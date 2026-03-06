variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "prod"
}

variable "instance_type" {
  description = "EC2 instance type. t3.xlarge (4 vCPU / 16GB) matches current Hetzner specs."
  type        = string
  default     = "t3.xlarge"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 50
}

variable "data_volume_size_gb" {
  description = "Data EBS volume size in GB (Postgres data + ML models)"
  type        = number
  default     = 200
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key for EC2 access"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH. Restrict to your IP(s) in production."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "airflow_ui_cidrs" {
  description = "CIDR blocks allowed to access Airflow UI (port 8080). Use SSH tunnel instead for security."
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "Public subnet CIDR"
  type        = string
  default     = "10.0.1.0/24"
}

variable "availability_zone" {
  description = "Availability zone for subnet and EBS volumes"
  type        = string
  default     = "us-east-1a"
}

variable "backup_retention_days" {
  description = "Days to retain S3 backups before expiring"
  type        = number
  default     = 30
}
