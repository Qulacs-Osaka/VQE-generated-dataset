OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.2426600914262755) q[0];
rz(0.8446990105843413) q[0];
ry(-3.023270126963076) q[1];
rz(-0.07764280794947886) q[1];
ry(-1.1075903847518038) q[2];
rz(1.1977242585618002) q[2];
ry(-0.6902980214917034) q[3];
rz(-0.6724302667347003) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.8342610289035037) q[0];
rz(2.720347219210079) q[0];
ry(2.015810217645043) q[1];
rz(-2.7159944189128837) q[1];
ry(-0.8154351570992602) q[2];
rz(2.8691571782564704) q[2];
ry(2.6735894099097215) q[3];
rz(-0.6569311982102093) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.8075276722511019) q[0];
rz(-1.575122974618293) q[0];
ry(2.44190338594171) q[1];
rz(-1.3879377682361174) q[1];
ry(-0.9995432182207156) q[2];
rz(-0.3729744540369824) q[2];
ry(-0.8915260056505885) q[3];
rz(0.3605603773414314) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.28362505408087746) q[0];
rz(1.6953076330751287) q[0];
ry(-0.4171728820991229) q[1];
rz(0.8270180105781085) q[1];
ry(-1.413570867486321) q[2];
rz(2.36648514057582) q[2];
ry(1.0456363104261772) q[3];
rz(-1.7121669973914493) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.7325087339789923) q[0];
rz(-3.0604771199936582) q[0];
ry(-2.843105290109031) q[1];
rz(-2.682935226213342) q[1];
ry(0.11685700427454386) q[2];
rz(-2.694111387148228) q[2];
ry(-1.5887016041995787) q[3];
rz(-2.570401920421266) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.73940624770169) q[0];
rz(2.794349330687071) q[0];
ry(-0.9515827786019847) q[1];
rz(1.637077307358851) q[1];
ry(-1.856485006981333) q[2];
rz(-0.23859846903455983) q[2];
ry(1.8236502339574034) q[3];
rz(-0.7663468468588005) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.4465583088837557) q[0];
rz(1.6000989983878302) q[0];
ry(2.930928072466377) q[1];
rz(1.098688013335332) q[1];
ry(-2.8831658703585776) q[2];
rz(-2.4109373902770628) q[2];
ry(-0.8234870738650959) q[3];
rz(2.132880967707965) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.957485217234168) q[0];
rz(-1.420182552541233) q[0];
ry(-0.5722845884529288) q[1];
rz(1.6084379847019425) q[1];
ry(2.193845484396353) q[2];
rz(-0.49319543959560264) q[2];
ry(1.6186688967565113) q[3];
rz(-2.906304907974753) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.1618467470751677) q[0];
rz(-0.0013020205483025296) q[0];
ry(0.1535620594835203) q[1];
rz(-0.2922063076360904) q[1];
ry(1.790390320822768) q[2];
rz(-1.8450893767151708) q[2];
ry(-0.04955013200125613) q[3];
rz(2.9911939722219723) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.8509364556715822) q[0];
rz(0.29652444676416323) q[0];
ry(-1.691690697228431) q[1];
rz(0.1620420056653479) q[1];
ry(-0.9766141581320307) q[2];
rz(-2.153478398047571) q[2];
ry(0.4254774897302491) q[3];
rz(-0.650252719621559) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.779551383420023) q[0];
rz(-2.904152555542815) q[0];
ry(-0.44650506869828616) q[1];
rz(0.19318048081835562) q[1];
ry(-2.455678744572948) q[2];
rz(3.027445151601623) q[2];
ry(0.8513144212204481) q[3];
rz(1.5358946132958178) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.3106068896343448) q[0];
rz(1.7790038523226632) q[0];
ry(0.8327110115469132) q[1];
rz(-2.1640975510337754) q[1];
ry(-2.014612346599475) q[2];
rz(1.6173481656651922) q[2];
ry(-2.1249115140255714) q[3];
rz(-1.2130387267543528) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.5284608165368787) q[0];
rz(-2.417866992224968) q[0];
ry(1.9105059193623268) q[1];
rz(2.2897501320306177) q[1];
ry(1.029897950596753) q[2];
rz(-0.22709204396588406) q[2];
ry(-0.13897522614979227) q[3];
rz(0.5611013254242501) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.1272075296942052) q[0];
rz(0.41921766956799666) q[0];
ry(0.7328771127235015) q[1];
rz(2.109078679409066) q[1];
ry(-1.8307014903908083) q[2];
rz(1.1513965956532826) q[2];
ry(-2.4824943651071947) q[3];
rz(0.5314303786799692) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.14326410123815148) q[0];
rz(-2.155434618373549) q[0];
ry(-2.7131529885376633) q[1];
rz(1.279427138295552) q[1];
ry(-1.9250140264625557) q[2];
rz(3.095081871496325) q[2];
ry(-3.0527445863981977) q[3];
rz(0.6189229148701576) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.2671748313759954) q[0];
rz(2.5321020587392518) q[0];
ry(-2.1494098640185326) q[1];
rz(-0.43562077626581774) q[1];
ry(-2.959513535359802) q[2];
rz(-0.24134122877767566) q[2];
ry(-0.600196320280703) q[3];
rz(-2.2992440665479785) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2470696012662537) q[0];
rz(0.4922883464847532) q[0];
ry(-1.2706809600628794) q[1];
rz(-3.112040523787365) q[1];
ry(0.4850201451633893) q[2];
rz(2.6184821897780193) q[2];
ry(0.3694624265904574) q[3];
rz(1.20033254612333) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.4867417099879816) q[0];
rz(-1.5207500388820747) q[0];
ry(0.25404420466986366) q[1];
rz(-0.9935117822985787) q[1];
ry(0.20787498451159095) q[2];
rz(2.6631457239516325) q[2];
ry(0.22587940814343976) q[3];
rz(2.849870894522335) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.043863485244262534) q[0];
rz(-0.333072415311774) q[0];
ry(0.34610548340139063) q[1];
rz(-2.6628366610023932) q[1];
ry(-1.4849866470671527) q[2];
rz(3.1192164814581966) q[2];
ry(0.37181088826189135) q[3];
rz(0.37806580540184154) q[3];