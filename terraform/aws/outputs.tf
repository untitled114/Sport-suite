output "elastic_ip" {
  description = "Static public IP — update DNS and deploy.sh DEPLOY_SERVER with this"
  value       = aws_eip.sport_suite.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.sport_suite.id
}

output "ssh_command" {
  description = "SSH into the instance"
  value       = "ssh sportsuite@${aws_eip.sport_suite.public_ip}"
}

output "airflow_tunnel" {
  description = "SSH tunnel for Airflow UI (no public port needed)"
  value       = "ssh -L 8080:localhost:8080 sportsuite@${aws_eip.sport_suite.public_ip}"
}

output "backups_bucket" {
  description = "S3 bucket for PostgreSQL backups"
  value       = aws_s3_bucket.backups.bucket
}

output "models_bucket" {
  description = "S3 bucket for ML model .pkl files"
  value       = aws_s3_bucket.models.bucket
}
