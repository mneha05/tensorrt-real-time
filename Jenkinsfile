pipeline {
  agent any
  stages {
    stage('Syntax') {
      steps {
        sh 'python3 -m py_compile trt_infer.py'
      }
    }
    stage('GPU smoke test') {
      when { expression { return fileExists('/usr/local/cuda') } }
      steps {
        sh 'python3 -c "import tensorrt; print(tensorrt.__version__)"'
      }
    }
  }
}
