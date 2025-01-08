class DisableBarrier:
    def __init__(self, delegate):
        self.delegate = delegate

    def __sfinv_marshal__(self, inv_capsule, ignored_resource_barrier):
        self.delegate.__sfinv_marshal__(inv_capsule, 3) # 3 == ProgramResourceBarrier::NONE
