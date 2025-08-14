import praw
import pymongo
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for Reddit scraping parameters"""
    posts_per_subreddit: int = 300
    comments_per_post: int = 30
    rate_limit_delay: int = 3
    max_retries: int = 3
    days_back: int = 90


class RedditMentalHealthScraper:
    def __init__(self):
        load_dotenv()
        self.config = ScrapingConfig()
        self._setup_mongodb()
        self._setup_reddit_client()

        # ETHICAL SUBREDDIT SELECTION - Removed SuicideWatch for safety
        self.subreddits = [
            "mentalhealth",  # General mental health discussions
            "depression",  # Depression support (moderated)
            "anxiety",  # Anxiety support
            "therapy",  # Therapy experiences
            "wellness",  # Wellness and self-care
            "ADHD",  # ADHD support
            "bipolar",  # Bipolar support
            "selfcare"  # Self-care strategies
        ]

        # Crisis-related keywords to flag for special handling
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'self harm',
            'cutting', 'overdose', 'no point', 'better off dead'
        ]

    def _setup_mongodb(self):
        """Setup MongoDB connection with error handling"""
        try:
            self.mongo_client = pymongo.MongoClient(
                os.getenv("MONGODB_CS"),
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.mongo_client.server_info()
            self.db = self.mongo_client["reddit_mental_health"]
            self.posts_collection = self.db["posts"]
            self.comments_collection = self.db["comments"]
            self.metadata_collection = self.db["scraping_metadata"]
            logger.info("MongoDB connection established successfully")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def _setup_reddit_client(self):
        """Setup Reddit API client with proper authentication"""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT"),
                ratelimit_seconds=600  # Handle rate limits gracefully
            )
            # Test authentication
            logger.info(f"Reddit client authenticated as: {self.reddit.user.me()}")
        except Exception as e:
            logger.error(f"Reddit authentication failed: {e}")
            raise

    def _anonymize_username(self, username: str) -> str:
        """Create anonymous hash of username for privacy"""
        if not username or username in ['[deleted]', None]:
            return 'anonymous'
        return hashlib.sha256(f"{username}_salt_key".encode()).hexdigest()[:12]

    def _detect_crisis_content(self, text: str) -> bool:
        """Detect crisis-related content that needs special handling"""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)

    def _sanitize_text(self, text: str) -> str:
        """Remove personally identifiable information from text"""
        if not text:
            return ""

        # Remove potential usernames (@username)
        text = re.sub(r'@\w+', '[USERNAME]', text)
        # Remove potential email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # Remove potential phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]',
                      text)

        return text

    def _get_posts_from_subreddit(self, subreddit_name: str) -> List[Dict]:
        """Fetch posts using only PRAW (no Pushshift)"""
        posts_data = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get posts from multiple time periods for better coverage
            post_generators = [
                subreddit.hot(limit=self.config.posts_per_subreddit // 3),
                subreddit.new(limit=self.config.posts_per_subreddit // 3),
                subreddit.top(time_filter='week', limit=self.config.posts_per_subreddit // 3)
            ]

            processed_ids = set()

            for generator in post_generators:
                for post in generator:
                    if post.id in processed_ids:
                        continue
                    processed_ids.add(post.id)

                    # Skip if already in database
                    if self.posts_collection.find_one({"reddit_id": post.id}):
                        continue

                    # Skip removed/deleted posts
                    if post.selftext in ['[removed]', '[deleted]', '']:
                        continue

                    # Sanitize and check for crisis content
                    sanitized_title = self._sanitize_text(post.title)
                    sanitized_body = self._sanitize_text(post.selftext)

                    is_crisis = self._detect_crisis_content(post.title + " " + post.selftext)

                    post_data = {
                        "reddit_id": post.id,
                        "subreddit": subreddit_name,
                        "title": sanitized_title,
                        "body": sanitized_body,
                        "author_hash": self._anonymize_username(str(post.author)),
                        "score": post.score,
                        "upvote_ratio": post.upvote_ratio,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "created_datetime": datetime.utcfromtimestamp(post.created_utc),
                        "url": post.url if post.url != post.permalink else None,
                        "is_crisis_flagged": is_crisis,
                        "scraped_at": datetime.utcnow(),
                        "flair": post.link_flair_text
                    }
                    posts_data.append(post_data)

                    # Rate limiting
                    time.sleep(self.config.rate_limit_delay)

        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}: {e}")

        return posts_data

    def _get_comments_for_post(self, post_id: str, reddit_post_id: str) -> List[Dict]:
        """Fetch comments for a specific post with privacy protection"""
        comments_data = []
        try:
            submission = self.reddit.submission(id=reddit_post_id)
            submission.comments.replace_more(limit=0)

            # Get top-level comments first, then some replies
            all_comments = submission.comments.list()

            for i, comment in enumerate(all_comments[:self.config.comments_per_post]):
                if comment.body in ['[removed]', '[deleted]']:
                    continue

                sanitized_body = self._sanitize_text(comment.body)
                is_crisis = self._detect_crisis_content(comment.body)

                comment_data = {
                    "reddit_id": comment.id,
                    "post_id": post_id,  # Reference to our post document
                    "reddit_post_id": reddit_post_id,
                    "author_hash": self._anonymize_username(str(comment.author)),
                    "body": sanitized_body,
                    "score": comment.score,
                    "created_utc": comment.created_utc,
                    "created_datetime": datetime.utcfromtimestamp(comment.created_utc),
                    "is_crisis_flagged": is_crisis,
                    "scraped_at": datetime.utcnow(),
                    "depth": 0 if comment.parent_id.startswith('t3_') else 1  # 0 for top-level, 1 for reply
                }
                comments_data.append(comment_data)

                # Small delay between comments
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error fetching comments for post {reddit_post_id}: {e}")

        return comments_data

    def _store_posts_and_comments(self, posts_data: List[Dict], subreddit_name: str):
        """Store posts and their comments in MongoDB"""
        for post_data in posts_data:
            try:
                # Insert post
                post_result = self.posts_collection.insert_one(post_data)
                post_object_id = post_result.inserted_id

                # Get comments for this post
                comments_data = self._get_comments_for_post(
                    str(post_object_id),
                    post_data['reddit_id']
                )

                # Insert comments if any
                if comments_data:
                    self.comments_collection.insert_many(comments_data)
                    logger.info(f"Stored post {post_data['reddit_id']} with {len(comments_data)} comments")
                else:
                    logger.info(f"Stored post {post_data['reddit_id']} with no comments")

                # Rate limiting between posts
                time.sleep(self.config.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error storing post {post_data.get('reddit_id', 'unknown')}: {e}")

    def _update_scraping_metadata(self, subreddit_name: str, posts_count: int, comments_count: int):
        """Track scraping statistics"""
        metadata = {
            "subreddit": subreddit_name,
            "scraping_date": datetime.utcnow(),
            "posts_scraped": posts_count,
            "comments_scraped": comments_count,
            "status": "completed"
        }
        self.metadata_collection.insert_one(metadata)

    def scrape_subreddit(self, subreddit_name: str):
        """Scrape a single subreddit with full error handling"""
        logger.info(f"Starting to scrape r/{subreddit_name}")

        try:
            # Get posts
            posts_data = self._get_posts_from_subreddit(subreddit_name)

            if not posts_data:
                logger.warning(f"No new posts found for r/{subreddit_name}")
                return

            # Store posts and comments
            initial_comment_count = self.comments_collection.count_documents({})
            self._store_posts_and_comments(posts_data, subreddit_name)
            final_comment_count = self.comments_collection.count_documents({})

            comments_added = final_comment_count - initial_comment_count

            # Update metadata
            self._update_scraping_metadata(subreddit_name, len(posts_data), comments_added)

            logger.info(f"Completed r/{subreddit_name}: {len(posts_data)} posts, {comments_added} comments")

        except Exception as e:
            logger.error(f"Failed to scrape r/{subreddit_name}: {e}")

    def scrape_all_subreddits(self):
        """Scrape all configured subreddits"""
        logger.info("Starting comprehensive mental health data scraping")

        total_start_time = time.time()

        for subreddit_name in self.subreddits:
            try:
                self.scrape_subreddit(subreddit_name)
                # Longer delay between subreddits
                time.sleep(10)
            except Exception as e:
                logger.error(f"Critical error with r/{subreddit_name}: {e}")
                continue

        total_time = time.time() - total_start_time
        logger.info(f"Scraping completed in {total_time:.2f} seconds")

        # Print summary statistics
        self.print_summary()

    def print_summary(self):
        """Print scraping summary statistics"""
        total_posts = self.posts_collection.count_documents({})
        total_comments = self.comments_collection.count_documents({})
        crisis_posts = self.posts_collection.count_documents({"is_crisis_flagged": True})
        crisis_comments = self.comments_collection.count_documents({"is_crisis_flagged": True})

        print("\n" + "=" * 50)
        print("SCRAPING SUMMARY")
        print("=" * 50)
        print(f"Total Posts: {total_posts}")
        print(f"Total Comments: {total_comments}")
        print(f"Crisis-Flagged Posts: {crisis_posts}")
        print(f"Crisis-Flagged Comments: {crisis_comments}")
        print(f"Subreddits Processed: {len(self.subreddits)}")
        print("=" * 50)

    def close_connections(self):
        """Clean up database connections"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logger.info("Database connections closed")


def main():
    scraper = RedditMentalHealthScraper()

    try:
        scraper.scrape_all_subreddits()
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        scraper.close_connections()


if __name__ == "__main__":
    main()