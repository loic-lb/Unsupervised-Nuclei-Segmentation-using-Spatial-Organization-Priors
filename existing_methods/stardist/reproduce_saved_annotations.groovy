def path = buildFilePath(PROJECT_BASE_DIR, 'annotations')
def annotations = null
new File(path).withObjectInputStream {
    annotations = it.readObject()
}
addObjects(annotations)
print 'Added ' + annotations