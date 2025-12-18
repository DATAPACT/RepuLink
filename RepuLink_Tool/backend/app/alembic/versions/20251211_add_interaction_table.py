"""Add interaction table

Revision ID: 20251211_add_interaction_table
Revises: d98dd8ec85a3
Create Date: 2025-12-11 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import sqlmodel.sql.sqltypes

# revision identifiers, used by Alembic.
revision = "20251211_add_interaction_table"
down_revision = "d98dd8ec85a3"
branch_labels = None
depends_on = None


def upgrade():
    # Ensure uuid-ossp extension is available
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "interaction",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("initiator_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=False), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=False), nullable=False),
        sa.ForeignKeyConstraint(["initiator_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["target_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("interaction")
