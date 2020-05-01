import os
import pymongo
from bson.objectid import ObjectId

class DbConnection():
  def __init__(self):
    self.db = pymongo.MongoClient(os.getenv("MONGO_URL"))[os.getenv("MONGO_DB")]

  def insert_identity(self, identity):
    result = self.db.identities.insert_one(identity)
    return result.inserted_id

  def get_identities(self, opt=None):
    return self.db.identities.find(opt)

  def get_logtimes(self, opt):
    return self.db.logtimes.find(opt)

  def insert_logtime(self, logtime):
    result = self.db.logtimes.insert_one(logtime)
    return result.inserted_id

  def set_logtime(self, opt, update):
    self.db.logtimes.update_one(opt, {"$set": update})

  def to_objectid(self, id):
    return ObjectId(id)