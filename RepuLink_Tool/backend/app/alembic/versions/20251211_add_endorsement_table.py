"""Add endorsement table

Revision ID: 20251211_add_endorsement_table
Revises: 20251211_add_rating_table
Create Date: 2025-12-11 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel.sql.sqltypes

# revision identifiers, used by Alembic.
revision = "20251211_add_endorsement_table"
down_revision = "20251211_add_rating_table"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "endorsement",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("endorser_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("endorsed_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=False), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=False), nullable=False),
        sa.ForeignKeyConstraint(["endorser_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["endorsed_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("endorser_id", "endorsed_id", name="uq_endorsement_endorser_endorsed"),
    )
    op.create_index(
        "ix_endorsement_endorser_id",
        "endorsement",
        ["endorser_id"],
        unique=False,
    )
    op.create_index(
        "ix_endorsement_endorsed_id",
        "endorsement",
        ["endorsed_id"],
        unique=False,
    )


def downgrade():
    op.drop_table("endorsement")
