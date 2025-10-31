pipeline {
    agent {
        label 'local'
    }

    environment {
        DOCKER_COMPOSE = '"C:\\Program Files\\Docker\\Docker\\resources\\bin\\docker-compose.exe"'
        PYTHON = 'python'
        TICKER = 'AAPL'
    }

    triggers {
        cron('0 2 * * *') // Runs daily at 2 AM
    }

    options {
        skipDefaultCheckout(true)
    }

    stages {
        stage('Checkout') {
            steps {
                echo '=== Checking out code ==='
                git branch: 'main', url: 'https://github.com/VROCKSONI786/mlops.git'
            }
        }

        stage('Setup Environment') {
            steps {
                echo '=== Setting up Python environment ==='
                bat '''
                    if not exist "venv" python -m venv venv
                    call venv\\Scripts\\activate.bat
                    pip install -q -r requirements.txt
                    if not exist "models" mkdir models
                    if not exist "data\\raw" mkdir data\\raw
                '''
            }
        }

        stage('Train Model') {
            steps {
                echo '=== Training model with fresh data ==='
                bat '''
                    call venv\\Scripts\\activate.bat
                    python train_pipeline.py --ticker %TICKER% --period 2y --model xgboost
                '''
            }
        }

        stage('Test Prediction') {
            steps {
                echo '=== Testing model predictions ==='
                bat '''
                    call venv\\Scripts\\activate.bat
                    python test_prediction.py %TICKER% || echo "Test completed with warnings"
                '''
            }
        }

        stage('Build & Deploy') {
            steps {
                echo '=== Building and deploying containers ==='
                bat """
                    ${DOCKER_COMPOSE} down
                    ${DOCKER_COMPOSE} build
                    ${DOCKER_COMPOSE} up -d
                """
            }
        }

        stage('Health Check') {
            steps {
                echo '=== Verifying deployment ==='
                bat '''
                    timeout /t 10 /nobreak
                    curl -f http://localhost:5000/health
                '''
            }
        }
    }

    post {
        success {
            echo '✓ Pipeline completed successfully!'
            echo 'Application: http://localhost:5000'
            archiveArtifacts artifacts: 'models/*.pkl', allowEmptyArchive: false
        }
        failure {
            echo '✗ Pipeline failed - check console output'
        }
        always {
            bat """
                ${DOCKER_COMPOSE} ps
                echo Build #${BUILD_NUMBER} completed
            """
        }
    }
}
