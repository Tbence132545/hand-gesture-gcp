variable "gcp_project_id" {
  type = string
}
variable "gcp_region" {
  type    = string
  default = "us-central1"
}
variable "service_name" {
  type    = string
  default = "gesture-recognition"
}
variable "container_image_url" {
  type = string
}
variable "user_email" {
  type = string
}
variable "vertex_endpoint_id" {
  description = "The ID of the manually created Vertex AI Endpoint."
  type        = string
}