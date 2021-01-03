# -*- coding: utf-8 -*-
# from requests.auth import HTTPBasicAuth
from config import db_creds

class Db:
    def __init__(self):
        self.db = None

    def get_db(self):
        if self.db is None:
            import pymysql
            self.db = pymysql.connect(local_infile=True, **db_creds)
            # import psycopg2
            # self.db = psycopg2.connect(**creds)
        return self.db
