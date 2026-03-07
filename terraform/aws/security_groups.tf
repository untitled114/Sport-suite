resource "aws_security_group" "sport_suite" {
  name        = "sport-suite-sg"
  description = "Sport-Suite EC2: SSH + Airflow UI + full outbound"
  vpc_id      = aws_vpc.main.id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Airflow UI — only open if cidrs provided, otherwise use SSH tunnel:
  #   ssh -L 8080:localhost:8080 sportsuite@<elastic-ip>
  dynamic "ingress" {
    for_each = length(var.airflow_ui_cidrs) > 0 ? [1] : []
    content {
      description = "Airflow UI (IPv4)"
      from_port   = 8080
      to_port     = 8080
      protocol    = "tcp"
      cidr_blocks = var.airflow_ui_cidrs
    }
  }

  dynamic "ingress" {
    for_each = length(var.airflow_ui_ipv6_cidrs) > 0 ? [1] : []
    content {
      description      = "Airflow UI (IPv6)"
      from_port        = 8080
      to_port          = 8080
      protocol         = "tcp"
      ipv6_cidr_blocks = var.airflow_ui_ipv6_cidrs
    }
  }

  # All outbound (ESPN API, BettingPros API, Discord, GitHub, package installs)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "sport-suite-sg" }
}
