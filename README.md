**Installation**

It is recommended to use virtual environment to ensure no problem with package version:

1. python3 -m venv venv
2. source venv/bin/activate


_Steps: _

1. Clone repo: git clone https://github.com/clovaai/deep-text-recognition-benchmark
2. Install: pip install -r requirements.txt
3. Run: python3 app.py
4. Test: curl -X POST -F "file=@demo_1.jpg" -F "language=en" http://localhost:5000/upload
