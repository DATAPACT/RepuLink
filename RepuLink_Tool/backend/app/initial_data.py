import logging

from sqlmodel import Session

from app.core.db import engine, init_db
from app import crud
from app.core.config import settings
from app.models import UserCreate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    with Session(engine) as session:
        init_db(session)
        
        # Create additional test users for interaction testing
        test_users = [
            UserCreate(
                email="alice@example.com",
                password="test123456",
                full_name="Alice Johnson",
            ),
            UserCreate(
                email="bob@example.com",
                password="test123456",
                full_name="Bob Smith",
            ),
            UserCreate(
                email="carol@example.com",
                password="test123456",
                full_name="Carol Williams",
            ),
            UserCreate(
                email="david@example.com",
                password="test123456",
                full_name="David Brown",
            ),
        ]
        
        for user_create in test_users:
            existing_user = session.query(crud.User).filter(
                crud.User.email == user_create.email
            ).first()
            if not existing_user:
                crud.create_user(session=session, user_create=user_create)
                logger.info(f"Created test user: {user_create.email}")


def main() -> None:
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
