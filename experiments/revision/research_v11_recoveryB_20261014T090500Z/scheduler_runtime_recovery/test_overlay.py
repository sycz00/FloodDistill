import unittest
from unittest.mock import patch
import worker11 as w

class OccupancyTests(unittest.TestCase):
    def row(self,**kw):
        return dict(known=True,host_threads=48,safe_threads=40,external_reserved_threads=32,free_ram_gib=100,load1=55,**kw)
    def test_localized_load_with_verified_spare_cores(self):
        self.assertTrue(w.availability(self.row(cpu_busy_verified=True,cpu_busy_cores=18),{'threads':2},2)[0])
    def test_real_busy_host_rejected(self):
        self.assertFalse(w.availability(self.row(cpu_busy_verified=True,cpu_busy_cores=39),{'threads':2},2)[0])
    def test_unknown_occupancy_uses_safe_load_fallback(self):
        self.assertFalse(w.availability(self.row(cpu_busy_verified=False),{'threads':2},2)[0])
        row=self.row(cpu_busy_verified=False);row['load1']=20
        self.assertTrue(w.availability(row,{'threads':2},2)[0])
    def test_configured_reservations_remain_independent_hard_gate(self):
        row=self.row(cpu_busy_verified=True,cpu_busy_cores=0);row['external_reserved_threads']=38
        self.assertFalse(w.availability(row,{'threads':2},2)[0])
    def test_maximum_of_two_short_samples(self):
        values=iter([dict(total=0,idle=0,cpus=10),dict(total=100,idle=80,cpus=10),dict(total=200,idle=140,cpus=10)])
        self.assertEqual(w.sampled_busy_cores(lambda:next(values),lambda _:None),(4.,[2.,4.]))
    def test_iowait_is_busy_and_guest_not_double_counted(self):
        with patch.object(w.Path,'read_text',return_value='cpu 10 0 0 80 10 0 0 0 5 0\ncpu0 0\ncpu1 0\n'):
            self.assertEqual(w.cpu_counter_sample(),dict(total=100,idle=80,cpus=2))
    def test_invalid_counter_delta_rejected(self):
        values=iter([dict(total=100,idle=80,cpus=10),dict(total=90,idle=85,cpus=10)])
        with self.assertRaises(ValueError):w.sampled_busy_cores(lambda:next(values),lambda _:None)

if __name__=='__main__':unittest.main()
