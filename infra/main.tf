terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.3.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# APIs and GCS Bucket
resource "google_project_service" "apis" {
  for_each = toset(["run.googleapis.com", "storage.googleapis.com", "iam.googleapis.com", "aiplatform.googleapis.com"])
  service  = each.key
  disable_on_destroy = false
}

resource "google_storage_bucket" "artifacts_bucket" {
  name                        = "${var.gcp_project_id}-${var.service_name}-artifacts"
  location                    = var.gcp_region
  uniform_bucket_level_access = true
  depends_on                  = [google_project_service.apis]
}


# Cloud Run Service and IAM 
resource "google_service_account" "cloud_run_sa" {
  account_id   = "${var.service_name}-runner"
  display_name = "SA for ${var.service_name} Cloud Run"
}

# Permission for Cloud Run to call the Vertex AI Endpoint
resource "google_project_iam_member" "vertex_invoker" {
  project = var.gcp_project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Permission for Cloud Run to read the scaler/encoder from GCS
resource "google_storage_bucket_iam_member" "gcs_reader" {
  bucket = google_storage_bucket.artifacts_bucket.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run_sa.email}"
}

# Deploy the Backend to Cloud Run
resource "google_cloud_run_v2_service" "main_service" {
  name     = var.service_name
  location = var.gcp_region
  
  template {
    containers {
      image = var.container_image_url
      ports { container_port = 8080 }
      
      env {
        name  = "GCP_PROJECT"
        value = var.gcp_project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.gcp_region
      }
      env {
        name  = "ARTIFACTS_BUCKET"
        value = google_storage_bucket.artifacts_bucket.name
      }
      # This now gets the ID from your terraform.tfvars file
      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = var.vertex_endpoint_id
      }
    }
    service_account = google_service_account.cloud_run_sa.email
  }
}
 
resource "google_cloud_run_v2_service_iam_member" "invoker" {
  project  = var.gcp_project_id
  location = var.gcp_region
  name     = google_cloud_run_v2_service.main_service.name
  role     = "roles/run.invoker"
  member   = var.user_email
}

output "cloud_run_service_url" {
  description = "The URL of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.main_service.uri
}

resource "google_cloud_run_v2_service_iam_member" "invoker_sa" {
  project  = var.gcp_project_id
  location = var.gcp_region
  name     = google_cloud_run_v2_service.main_service.name
  role     = "roles/run.invoker"
  # This line gives the service account permission to call the service
  member   = "serviceAccount:${google_service_account.cloud_run_sa.email}" 
}
