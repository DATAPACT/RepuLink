import uuid

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from datetime import datetime


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)  # type: ignore
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(ItemBase):
    title: str | None = Field(default=None, min_length=1, max_length=255)  # type: ignore


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


# Interaction models
class InteractionBase(SQLModel):
    message: str | None = Field(default=None, max_length=1024)
    status: str = Field(default="pending", max_length=32)


class InteractionCreate(InteractionBase):
    target_id: uuid.UUID


class InteractionPublic(InteractionBase):
    id: uuid.UUID
    initiator_id: uuid.UUID
    target_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Interaction(InteractionBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    initiator_id: uuid.UUID = Field(foreign_key="user.id", nullable=False)
    target_id: uuid.UUID = Field(foreign_key="user.id", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# Rating models
class RatingBase(SQLModel):
    rating: int = Field(ge=-5, le=5)
    comment: str | None = Field(default=None, max_length=1024)


class RatingCreate(RatingBase):
    pass


class RatingPublic(RatingBase):
    id: uuid.UUID
    interaction_id: uuid.UUID
    rater_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RatingWithRater(RatingPublic):
    """Rating with rater user information"""
    rater_email: str
    rater_full_name: str | None


class Rating(RatingBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    interaction_id: uuid.UUID = Field(foreign_key="interaction.id", nullable=False)
    rater_id: uuid.UUID = Field(foreign_key="user.id", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Endorsement models
class EndorsementBase(SQLModel):
    confidence: float = Field(ge=0.0, le=1.0)


class EndorsementCreate(EndorsementBase):
    endorsed_id: uuid.UUID


class EndorsementPublic(EndorsementBase):
    id: uuid.UUID
    endorser_id: uuid.UUID
    endorsed_id: uuid.UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EndorsementWithUser(EndorsementPublic):
    """Endorsement with user information"""
    user_email: str
    user_full_name: str | None


class Endorsement(EndorsementBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    endorser_id: uuid.UUID = Field(foreign_key="user.id", nullable=False, ondelete="CASCADE")
    endorsed_id: uuid.UUID = Field(foreign_key="user.id", nullable=False, ondelete="CASCADE")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    __table_args__ = (
        __import__("sqlalchemy").UniqueConstraint("endorser_id", "endorsed_id", name="uq_endorsement_endorser_endorsed"),
    )
