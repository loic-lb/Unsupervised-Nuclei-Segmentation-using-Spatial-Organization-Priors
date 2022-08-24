
def path = buildFilePath(PROJECT_BASE_DIR, 'annotations')
def annotations = getAnnotationObjects()
new File(path).withObjectOutputStream {
    it.writeObject(annotations)
}
print 'Done!'