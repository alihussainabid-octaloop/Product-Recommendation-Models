from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()


def generate_pass_hash(plain_pass: str):
    passhash = password_hash.hash(plain_pass)
    return passhash


print(generate_pass_hash("secret"))
