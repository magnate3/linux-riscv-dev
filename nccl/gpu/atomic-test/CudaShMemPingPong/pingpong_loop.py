import subprocess

output_file = '/users/sohamb/GHConsistencyTests_pingpong/output.txt'

for _ in range(100):
    result = subprocess.run(['./MP.out'], capture_output=True, text=True)
    with open(output_file, 'a') as f:
        f.write(result.stdout)