# Restore the exact recorded package versions into a project-local library.
library_path <- "renv/library"
dir.create(library_path,recursive=TRUE,showWarnings=FALSE)
.libPaths(c(normalizePath(library_path),.libPaths()))
if(!requireNamespace("renv",quietly=TRUE)) install.packages("renv",lib=library_path,repos="https://cloud.r-project.org")
renv::restore(lockfile="renv.lock",library=library_path,prompt=FALSE)
