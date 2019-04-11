
#include "tracking.hpp"

using Manager_t = VariadicTrackManager<Electron, Proton>;

template <typename _Tp>
void print(const _Tp* t, int expected_id, const std::string& access_method)
{
    auto w = 24;
    std::cout << "track " << expected_id << " (" << access_method << "). name : " << std::setw(15)
              << t->GetParticleName() << ",  id : " << t->m_track_id << ",  typeid : " << std::setw(w)
              << typeid(*t).name() << ", particle def typeid : " << std::setw(10)
              << typeid(*t->GetParticleDefinition()).name() << std::endl;
}

int main()
{
    Electron* electron = new Electron();
    Proton* proton = new Proton();
    Track* t1 = new Track();
    Track* t2 = new Track();
    Track* t3 = new Track();

    t1->m_pdef = electron;
    t1->m_track_id = 0;
    t2->m_pdef = proton;
    t2->m_track_id = 1;
    t3->m_pdef = proton;
    t3->m_track_id = 2;

    Manager_t* manager = new Manager_t;

    manager->PushTrack<Electron>(t1);
    manager->PushTrack(t2);
    manager->PushTrack<Proton>(t3);

    auto _t1 = manager->PopTrack<Electron>();
    auto _t2 = manager->PopTrack();
    auto _t3 = manager->PopTrack<Proton>();

    print(t1, 0, "pushed");
    print(t2, 1, "pushed");
    print(t3, 2, "pushed");
    print(_t1, 0, "popped");
    print(_t2, 1, "popped");
    print(_t3, 2, "popped");

    delete manager;
    delete t1;
    delete t2;
    delete t3;
    delete electron;
    delete proton;
}