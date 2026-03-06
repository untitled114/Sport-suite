resource "aws_key_pair" "deployer" {
  key_name   = "sport-suite-deployer"
  public_key = file(var.ssh_public_key_path)
}

resource "aws_instance" "sport_suite" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.sport_suite.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  # Root volume — OS + code + venv
  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gb
    delete_on_termination = true
    encrypted             = true

    tags = { Name = "sport-suite-root" }
  }

  user_data = file("${path.module}/scripts/user_data.sh")

  # Prevent accidental termination
  disable_api_termination = true

  tags = { Name = "sport-suite" }

  lifecycle {
    # AMI updates should be handled manually (avoid re-provisioning)
    ignore_changes = [ami, user_data]
  }
}

# Separate data volume for PostgreSQL data + ML models
# Mounted at /data — survives instance replacement
resource "aws_ebs_volume" "data" {
  availability_zone = var.availability_zone
  size              = var.data_volume_size_gb
  type              = "gp3"
  encrypted         = true

  tags = { Name = "sport-suite-data" }
}

resource "aws_volume_attachment" "data" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.data.id
  instance_id  = aws_instance.sport_suite.id
  force_detach = false
}

# Static IP — stays the same across instance stops/starts
resource "aws_eip" "sport_suite" {
  instance = aws_instance.sport_suite.id
  domain   = "vpc"

  tags = { Name = "sport-suite-eip" }

  depends_on = [aws_internet_gateway.main]
}
