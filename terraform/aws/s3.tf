resource "aws_s3_bucket" "backups" {
  bucket = "sport-suite-backups-${data.aws_caller_identity.current.account_id}"

  tags = { Name = "sport-suite-backups", Purpose = "postgres-backups" }
}

resource "aws_s3_bucket" "models" {
  bucket = "sport-suite-models-${data.aws_caller_identity.current.account_id}"

  tags = { Name = "sport-suite-models", Purpose = "ml-model-pkl-files" }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Encrypt at rest
resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Expire old backups automatically
resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "expire-old-backups"
    status = "Enabled"

    filter { prefix = "" }

    expiration {
      days = var.backup_retention_days
    }
  }
}

# Current AWS account ID (used for unique bucket naming)
data "aws_caller_identity" "current" {}
