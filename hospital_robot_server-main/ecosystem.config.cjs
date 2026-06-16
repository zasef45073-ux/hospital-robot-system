module.exports = {
  apps: [
    {
      name: 'hospital-robot',
      script: 'python3',
      args: 'app.py',
      cwd: '/home/user/hospital_robot_server',
      watch: false,
      instances: 1,
      exec_mode: 'fork',
      env: {
        FLASK_ENV: 'development',
        SECRET_KEY: 'hospital-robot-secret-2024',
        APP_USER: 'admin',
        APP_PASS: 'admin1234'
      }
    }
  ]
}
