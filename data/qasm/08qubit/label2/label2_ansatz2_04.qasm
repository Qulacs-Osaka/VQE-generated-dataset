OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(1.5522975923683415) q[0];
rz(0.17001094813831744) q[0];
ry(1.647849438840379) q[1];
rz(-0.06782802550990992) q[1];
ry(-3.1090209250502845) q[2];
rz(-3.0299724416554317) q[2];
ry(-3.1340567313566257) q[3];
rz(-0.4631477680173548) q[3];
ry(3.1415904912891666) q[4];
rz(1.9399567236162953) q[4];
ry(-5.433619270297772e-07) q[5];
rz(-0.1923896158447622) q[5];
ry(1.557506807881026e-05) q[6];
rz(-1.9646290457212627) q[6];
ry(-3.1415866440916034) q[7];
rz(-2.9925225947945306) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-2.7420864588897818) q[0];
rz(2.719889166493852) q[0];
ry(1.1427000252914439) q[1];
rz(-2.2384164709850776) q[1];
ry(-2.297343039847117) q[2];
rz(2.2569682355166565) q[2];
ry(-3.0329146642713054) q[3];
rz(-2.856755668984174) q[3];
ry(1.5710998298346235) q[4];
rz(-3.141120934955111) q[4];
ry(-3.1415876564694045) q[5];
rz(1.778857287005376) q[5];
ry(-3.1415307717618406) q[6];
rz(-1.4834320621895287) q[6];
ry(3.1414061411543397) q[7];
rz(2.206410621724727) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.645321108081134e-06) q[0];
rz(-1.2832240146210054) q[0];
ry(-3.141590753305368) q[1];
rz(-1.6334721798786154) q[1];
ry(5.40299219773388e-05) q[2];
rz(-0.699830491317599) q[2];
ry(-2.419638265117046e-05) q[3];
rz(-1.4693290605415843) q[3];
ry(-1.570809000745963) q[4];
rz(1.5772466500638656) q[4];
ry(-0.046645256588725204) q[5];
rz(2.6712504443943246) q[5];
ry(1.4670760535055258) q[6];
rz(1.2965739038178943) q[6];
ry(-1.9070902530948786e-05) q[7];
rz(-1.586138773720788) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.1415821667609167) q[0];
rz(-1.8855454231387414) q[0];
ry(-2.1124989691696187e-05) q[1];
rz(2.534452315879947) q[1];
ry(-3.1415327896137053) q[2];
rz(1.5186501251370768) q[2];
ry(5.576499748606294e-06) q[3];
rz(0.8409649683095335) q[3];
ry(3.1415756050070085) q[4];
rz(0.3242577412542969) q[4];
ry(3.0886228844668513) q[5];
rz(-1.3251250898114102) q[5];
ry(0.536872039719567) q[6];
rz(-2.390823029926567) q[6];
ry(1.5707997193390077) q[7];
rz(3.1413092319368063) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5624797957242054) q[0];
rz(1.670406907609743) q[0];
ry(-1.578967171417842) q[1];
rz(1.6618969156315397) q[1];
ry(-1.5658771922555026) q[2];
rz(1.4629159523240878) q[2];
ry(1.5716477375074662) q[3];
rz(-0.4644967116624985) q[3];
ry(-0.1213009419581212) q[4];
rz(1.351815598523018) q[4];
ry(1.216933760838644e-05) q[5];
rz(-0.9726653742797943) q[5];
ry(-3.1412775983852415) q[6];
rz(-1.1342659302014433) q[6];
ry(1.5706754820170383) q[7];
rz(-1.3880546430989649) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(2.6758002755607935) q[0];
rz(1.68306312575013) q[0];
ry(0.46529377890957524) q[1];
rz(1.5930297131640974) q[1];
ry(2.6658743828693487) q[2];
rz(-1.6890301690361562) q[2];
ry(1.6368119465890745) q[3];
rz(1.864771220418052) q[3];
ry(-1.5697432586993758) q[4];
rz(1.5750866540500965) q[4];
ry(-2.332929107673465e-05) q[5];
rz(-0.884275670034314) q[5];
ry(-1.989404142267237e-06) q[6];
rz(-3.030613337563258) q[6];
ry(-3.141590124776838) q[7];
rz(1.6680055639908937) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.034593810472422715) q[0];
rz(1.9053871328350884) q[0];
ry(0.007345526168347938) q[1];
rz(-1.6226993109546701) q[1];
ry(0.8043762655780577) q[2];
rz(2.2676056658081847) q[2];
ry(-3.016881778416778) q[3];
rz(-0.22432245363332015) q[3];
ry(-1.515773185184515) q[4];
rz(-2.7007150913514493) q[4];
ry(1.570785783830215) q[5];
rz(0.0006752885741176939) q[5];
ry(1.57125651373915) q[6];
rz(3.140284267524746) q[6];
ry(-1.5703515169694329) q[7];
rz(3.1408781938007193) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-5.431590024303108e-07) q[0];
rz(-1.4689712214632014) q[0];
ry(-3.141592539292396) q[1];
rz(-1.061102200838553) q[1];
ry(-0.0002690256371786206) q[2];
rz(1.316157977649035) q[2];
ry(-3.1413130029626117) q[3];
rz(-1.694035541264146) q[3];
ry(-3.140848455846315) q[4];
rz(-2.311674371427161) q[4];
ry(1.5711207965117668) q[5];
rz(-1.1833410475755084) q[5];
ry(-1.570501441912695) q[6];
rz(-1.1834813030347393) q[6];
ry(-1.570501810092595) q[7];
rz(-1.1834820219612359) q[7];