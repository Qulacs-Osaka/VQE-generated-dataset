OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.0113450624886533) q[0];
rz(1.8234197259134022) q[0];
ry(-2.385763682150514) q[1];
rz(1.4810527585864088) q[1];
ry(2.7840974446319375) q[2];
rz(-1.152530319049002) q[2];
ry(-2.0763517071310162) q[3];
rz(0.023017967402661377) q[3];
ry(1.8286551791045813) q[4];
rz(-1.599402482246421) q[4];
ry(-3.0853965452593375) q[5];
rz(1.3841834705664235) q[5];
ry(-0.17384453532780275) q[6];
rz(2.1861633231964883) q[6];
ry(2.3208823130726004) q[7];
rz(-3.041622263094779) q[7];
ry(-2.459927390467729) q[8];
rz(0.02502474881277131) q[8];
ry(2.244757082490851) q[9];
rz(1.4364954398553547) q[9];
ry(0.8694812562629021) q[10];
rz(-1.002905807547946) q[10];
ry(0.81528597343529) q[11];
rz(1.2381708787348238) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.48068794687555855) q[0];
rz(0.12546631061552033) q[0];
ry(-0.536757560431373) q[1];
rz(0.9914105122338132) q[1];
ry(-1.444334113706311) q[2];
rz(-2.6003746050902925) q[2];
ry(0.6851860075460142) q[3];
rz(-0.527366463505758) q[3];
ry(2.764530088926038) q[4];
rz(-2.277483994376312) q[4];
ry(2.5236022746382183) q[5];
rz(-2.6717055387673505) q[5];
ry(1.4966490876269296) q[6];
rz(2.6278407532110917) q[6];
ry(-2.4445282066550913) q[7];
rz(-0.7431828261034302) q[7];
ry(-1.7427682916064944) q[8];
rz(-1.212681434950741) q[8];
ry(0.9775267116444528) q[9];
rz(1.3085886999188316) q[9];
ry(-1.4506312562492074) q[10];
rz(2.119624697631843) q[10];
ry(-1.0590563671504003) q[11];
rz(2.5776992007207586) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.7796456276535659) q[0];
rz(1.2746553741934248) q[0];
ry(0.7236362386563657) q[1];
rz(-1.5957116491151133) q[1];
ry(-1.7929058028575506) q[2];
rz(-0.2682220833527542) q[2];
ry(2.0647520911006554) q[3];
rz(-0.6336097793819313) q[3];
ry(-0.6970898940678643) q[4];
rz(2.9611130146158953) q[4];
ry(-0.574366397020031) q[5];
rz(0.5846937920805957) q[5];
ry(-0.2535508011996441) q[6];
rz(-1.792254484024566) q[6];
ry(-1.0635239980220197) q[7];
rz(2.98030314800467) q[7];
ry(0.7616055425162669) q[8];
rz(-0.8448183909820512) q[8];
ry(-1.5959879543328035) q[9];
rz(0.20955776854161104) q[9];
ry(-0.49971824710681556) q[10];
rz(-2.247235619631656) q[10];
ry(0.2971453516209302) q[11];
rz(-1.3752526207898192) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.1000991500874244) q[0];
rz(-0.015847529207302813) q[0];
ry(2.5885564556671192) q[1];
rz(2.8387892887602355) q[1];
ry(-1.6740714462210853) q[2];
rz(0.8010380665001255) q[2];
ry(2.3457984905727574) q[3];
rz(2.6611913900252167) q[3];
ry(0.4457690347275305) q[4];
rz(0.5991674532372624) q[4];
ry(-1.192868843214665) q[5];
rz(-0.6577310468062763) q[5];
ry(0.4596367297389454) q[6];
rz(0.49537257982859106) q[6];
ry(-0.7137975392577793) q[7];
rz(-2.7086957850768005) q[7];
ry(-1.6935396732478754) q[8];
rz(0.602746250776597) q[8];
ry(2.90935227257246) q[9];
rz(1.040506031679467) q[9];
ry(1.5734829998779905) q[10];
rz(-1.312832017154457) q[10];
ry(-2.0076865630479066) q[11];
rz(-1.9215007582343553) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.2221677025801119) q[0];
rz(1.19869736070192) q[0];
ry(-2.4562354055092195) q[1];
rz(1.434405694273162) q[1];
ry(2.584178724068245) q[2];
rz(-1.1094104461346974) q[2];
ry(0.3702245338199131) q[3];
rz(1.594222483966166) q[3];
ry(-1.9991071322473732) q[4];
rz(2.2541676228405203) q[4];
ry(2.925807335125044) q[5];
rz(1.3395518848541446) q[5];
ry(1.3141037488594876) q[6];
rz(-3.1343969552424356) q[6];
ry(-0.7848651344696308) q[7];
rz(2.2768215145445594) q[7];
ry(1.1830102478258495) q[8];
rz(-1.7471381043780578) q[8];
ry(-2.180441578275703) q[9];
rz(-0.5532914257757581) q[9];
ry(0.6261429765104225) q[10];
rz(-1.3671906350084022) q[10];
ry(-0.2892471921546526) q[11];
rz(2.331854343020707) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.36874507051041494) q[0];
rz(1.3666323375035825) q[0];
ry(-1.5348171926259742) q[1];
rz(-2.7756352804752478) q[1];
ry(-2.003334223427924) q[2];
rz(1.2290678742962469) q[2];
ry(2.073362605766009) q[3];
rz(3.0727885903126952) q[3];
ry(-1.1077832424607887) q[4];
rz(-2.3322434973211843) q[4];
ry(-2.471486475674637) q[5];
rz(-0.9526165683965591) q[5];
ry(1.8354428540423906) q[6];
rz(2.8583553107690536) q[6];
ry(0.7412993984989871) q[7];
rz(-2.599367318579024) q[7];
ry(2.8816622183713223) q[8];
rz(1.7141188608020599) q[8];
ry(2.139874066072858) q[9];
rz(2.5698740569468224) q[9];
ry(-0.3180510978117495) q[10];
rz(2.903328505054712) q[10];
ry(2.175907745763544) q[11];
rz(1.5505023184409574) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.9132895428893484) q[0];
rz(-0.37964467823224807) q[0];
ry(-0.47210977008740207) q[1];
rz(-2.605710623081864) q[1];
ry(2.6197079619704895) q[2];
rz(-2.97161315664426) q[2];
ry(-1.1697640920135535) q[3];
rz(-0.42404876208161196) q[3];
ry(-1.0349726401903436) q[4];
rz(1.615223358523874) q[4];
ry(0.42128328538590853) q[5];
rz(-2.5149973665795513) q[5];
ry(0.4247930227236906) q[6];
rz(-1.3286230699691912) q[6];
ry(1.4391302565058597) q[7];
rz(-0.6398341505508531) q[7];
ry(-1.3787913922245256) q[8];
rz(2.6597597835299243) q[8];
ry(-2.4130471345009665) q[9];
rz(-2.715365214811209) q[9];
ry(1.1218720147149428) q[10];
rz(2.118184282154581) q[10];
ry(-1.143932493866413) q[11];
rz(-2.172150857663798) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.1144971038518658) q[0];
rz(-2.3027018188373467) q[0];
ry(-2.2917730409195207) q[1];
rz(-1.0179346992888583) q[1];
ry(0.5350468197193248) q[2];
rz(-2.3670271297841587) q[2];
ry(-2.2868634641310086) q[3];
rz(1.2634312462151271) q[3];
ry(-1.7230288324836427) q[4];
rz(2.207203843923896) q[4];
ry(1.6640244033496343) q[5];
rz(2.7117842065849116) q[5];
ry(0.8996312164319168) q[6];
rz(-0.13656210404477975) q[6];
ry(-1.7634931939542495) q[7];
rz(-2.2879669919855634) q[7];
ry(-2.8850556442954707) q[8];
rz(2.8856408616949842) q[8];
ry(2.756947812349867) q[9];
rz(-3.026548565227394) q[9];
ry(1.5210643132839092) q[10];
rz(-0.7677068085705965) q[10];
ry(-1.1604366479464954) q[11];
rz(-1.0498656221960943) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.658059029385715) q[0];
rz(2.4356305611282933) q[0];
ry(-2.316807493704588) q[1];
rz(3.024946296542897) q[1];
ry(2.654487425999958) q[2];
rz(-0.0022348188951082106) q[2];
ry(0.4951714385231068) q[3];
rz(-0.7742860960800817) q[3];
ry(-1.929077852164168) q[4];
rz(2.2442752454178243) q[4];
ry(2.8904946697893408) q[5];
rz(-1.7485534268616727) q[5];
ry(-2.65973311115373) q[6];
rz(2.294362871163236) q[6];
ry(0.8310271951222497) q[7];
rz(3.0786121980609042) q[7];
ry(2.5633678731494194) q[8];
rz(0.08569103340813644) q[8];
ry(-2.8812331747547613) q[9];
rz(-0.308931697266833) q[9];
ry(0.5075742725874549) q[10];
rz(-1.7565272785617763) q[10];
ry(0.44304146138094463) q[11];
rz(0.9411044539382244) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.17844179303626984) q[0];
rz(-2.4844615104175682) q[0];
ry(-1.6065666038466748) q[1];
rz(-2.845019248068026) q[1];
ry(-1.246699922601458) q[2];
rz(-3.1226564729084805) q[2];
ry(0.7033018767635103) q[3];
rz(1.5274636190121733) q[3];
ry(-2.196742581347455) q[4];
rz(-1.9314628970655772) q[4];
ry(0.5517920310526867) q[5];
rz(-1.003361743451733) q[5];
ry(-1.3344885627321839) q[6];
rz(-2.7704339256249835) q[6];
ry(-0.8020200957754273) q[7];
rz(0.9349796741547491) q[7];
ry(-0.8701204173145909) q[8];
rz(-0.7386487735706484) q[8];
ry(1.5826952699583727) q[9];
rz(1.001500718740819) q[9];
ry(-2.818967602780197) q[10];
rz(1.5479187852630565) q[10];
ry(-2.6234310272517183) q[11];
rz(-2.5255963084004356) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.9175259904415372) q[0];
rz(-1.6781324766653962) q[0];
ry(2.850696174544929) q[1];
rz(-1.2971183646678135) q[1];
ry(2.9283537537737314) q[2];
rz(-1.8536783189739274) q[2];
ry(2.8619677903939387) q[3];
rz(-1.7222622431748986) q[3];
ry(-0.8292026768978821) q[4];
rz(1.5330568526079207) q[4];
ry(-2.364687657085468) q[5];
rz(3.0085912836934066) q[5];
ry(2.285097077828444) q[6];
rz(-0.5035367737451095) q[6];
ry(2.3339531514860523) q[7];
rz(-0.12218212165383281) q[7];
ry(-1.0727888727245858) q[8];
rz(-2.7539040434979563) q[8];
ry(-1.0549664814548825) q[9];
rz(0.6887826550832914) q[9];
ry(2.4934634268892792) q[10];
rz(0.8367806060546873) q[10];
ry(0.7483625986880872) q[11];
rz(0.7810255129660604) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.2677299682543177) q[0];
rz(2.0786660316793326) q[0];
ry(-2.0851718784346156) q[1];
rz(1.908758628219716) q[1];
ry(2.8716720409549965) q[2];
rz(-2.9720313489498325) q[2];
ry(1.7499249002073025) q[3];
rz(-1.245995170682869) q[3];
ry(-0.4782090228017157) q[4];
rz(-2.4203181137352816) q[4];
ry(-1.8301391691459012) q[5];
rz(0.5888092747557137) q[5];
ry(-2.8261937551748004) q[6];
rz(2.1531860227162634) q[6];
ry(-0.9736447685659586) q[7];
rz(-2.086314088779349) q[7];
ry(2.0787473338384435) q[8];
rz(2.293722923871219) q[8];
ry(-0.19967479604440985) q[9];
rz(2.537107658469877) q[9];
ry(-2.4560704161504883) q[10];
rz(0.49393951559329263) q[10];
ry(-0.8304026819842489) q[11];
rz(2.166790352399844) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.9259274609915098) q[0];
rz(-2.9640295899097078) q[0];
ry(-0.3877926891701858) q[1];
rz(1.609164543036671) q[1];
ry(2.8298589358962594) q[2];
rz(-2.09624390427281) q[2];
ry(1.939281476953103) q[3];
rz(0.8931750689480883) q[3];
ry(2.566817231835118) q[4];
rz(-3.0513784051418935) q[4];
ry(-2.3554148984671164) q[5];
rz(2.129919024542313) q[5];
ry(2.5902746203669804) q[6];
rz(0.5429800955615187) q[6];
ry(1.143704523265191) q[7];
rz(-1.8097666135757642) q[7];
ry(-1.0562729020110124) q[8];
rz(-0.7007206715719858) q[8];
ry(-2.897842027994662) q[9];
rz(-2.285456155607115) q[9];
ry(-2.3168119618009944) q[10];
rz(-2.5974822093365275) q[10];
ry(-0.06383072621538133) q[11];
rz(-2.1370279228333304) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.3439788272844384) q[0];
rz(-0.8407181241131827) q[0];
ry(2.819820848904688) q[1];
rz(2.3858371059592813) q[1];
ry(-2.054687876516759) q[2];
rz(-1.2708774388327286) q[2];
ry(-0.6490989224728643) q[3];
rz(0.33890587117978765) q[3];
ry(-2.3318699932089526) q[4];
rz(1.694906404097597) q[4];
ry(-0.5614806133456991) q[5];
rz(-0.8816846827170739) q[5];
ry(-0.533865317669636) q[6];
rz(0.5477264538606763) q[6];
ry(2.6377489856789516) q[7];
rz(-2.623065062463081) q[7];
ry(-2.5762464178137403) q[8];
rz(0.2050566014199356) q[8];
ry(0.8264090459562956) q[9];
rz(-0.32061037532045433) q[9];
ry(-1.3649517230970305) q[10];
rz(1.0728950009828928) q[10];
ry(1.672274183222209) q[11];
rz(0.25069238377293335) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.691228202425311) q[0];
rz(-1.6086345634839212) q[0];
ry(-1.3842355264994572) q[1];
rz(1.416712032900187) q[1];
ry(1.019246301890604) q[2];
rz(-1.23381998400645) q[2];
ry(1.6357671502853535) q[3];
rz(-0.5365682893278382) q[3];
ry(-2.2749460225724603) q[4];
rz(1.4060110085235156) q[4];
ry(-1.2287629675523988) q[5];
rz(-2.2835808133118074) q[5];
ry(-0.9360398012135651) q[6];
rz(0.8912244221090013) q[6];
ry(0.4733141238694074) q[7];
rz(-1.342075155593652) q[7];
ry(-0.6129129697877442) q[8];
rz(-2.462311418239407) q[8];
ry(-2.57391035904061) q[9];
rz(3.090616714290569) q[9];
ry(-1.9591698646071096) q[10];
rz(0.6152304699001663) q[10];
ry(0.6131879008937533) q[11];
rz(-0.6904678925402152) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.7672355083791977) q[0];
rz(2.9558162766420115) q[0];
ry(-2.2035397554400054) q[1];
rz(-0.4197495849532534) q[1];
ry(1.7282616182807506) q[2];
rz(-2.1554158911325887) q[2];
ry(1.4812265504149051) q[3];
rz(1.2662340808697015) q[3];
ry(0.22584700313438152) q[4];
rz(-1.2640301886742735) q[4];
ry(-1.0296674313385805) q[5];
rz(2.409755609297132) q[5];
ry(-2.4603319126256715) q[6];
rz(1.548277920706516) q[6];
ry(2.575498210804627) q[7];
rz(-1.6973486888261373) q[7];
ry(1.8420010247416) q[8];
rz(0.9774395418857226) q[8];
ry(0.6614820644725716) q[9];
rz(2.660524539784547) q[9];
ry(2.358284750581996) q[10];
rz(-1.6513230290716194) q[10];
ry(0.2864115532310487) q[11];
rz(-0.6464448577655536) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.178509539023173) q[0];
rz(2.7153754033250497) q[0];
ry(-0.9689658492430491) q[1];
rz(-0.24141407349638122) q[1];
ry(2.163916420044538) q[2];
rz(-1.380365173769996) q[2];
ry(-2.4297204226083875) q[3];
rz(2.059080783444119) q[3];
ry(-2.775311361720061) q[4];
rz(2.033467177979438) q[4];
ry(-2.016057565623195) q[5];
rz(0.6704451184642908) q[5];
ry(0.546671100293211) q[6];
rz(-1.0732744664624292) q[6];
ry(-0.8635491984290793) q[7];
rz(-0.7126692047633525) q[7];
ry(2.8420799711664415) q[8];
rz(-1.303156461740735) q[8];
ry(1.4615441508542129) q[9];
rz(2.711218967404979) q[9];
ry(-1.490641860893013) q[10];
rz(-2.2870562690632794) q[10];
ry(1.3932287041568128) q[11];
rz(-0.6774734067381889) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.7239033280243348) q[0];
rz(-0.8586958602736325) q[0];
ry(0.9993428096760333) q[1];
rz(-1.4701012061364964) q[1];
ry(-2.49576055651908) q[2];
rz(0.5372706303181376) q[2];
ry(2.5484394084030293) q[3];
rz(0.5734966722338345) q[3];
ry(1.2870582546492129) q[4];
rz(0.12017085121971505) q[4];
ry(1.6755669760311236) q[5];
rz(0.8884033755351404) q[5];
ry(2.1156519601072628) q[6];
rz(0.03538505245960308) q[6];
ry(2.0466651352366427) q[7];
rz(-2.5742924887225285) q[7];
ry(2.3737293167342326) q[8];
rz(1.6874397518592605) q[8];
ry(1.101613434839706) q[9];
rz(2.553122562065022) q[9];
ry(-2.8865495859448846) q[10];
rz(1.5021670002512924) q[10];
ry(-1.7021110956397283) q[11];
rz(-2.413003076190028) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.7949468729758825) q[0];
rz(-1.5169486486936332) q[0];
ry(-2.3791232467299976) q[1];
rz(-1.4963160781936615) q[1];
ry(-0.23823183114438679) q[2];
rz(-0.4959941673118937) q[2];
ry(2.830580403483801) q[3];
rz(0.8080782958814484) q[3];
ry(2.2438365691859046) q[4];
rz(-0.9969345958091598) q[4];
ry(-1.5820722980565394) q[5];
rz(0.9898827307505097) q[5];
ry(1.8009044681953927) q[6];
rz(-0.7501110268867135) q[6];
ry(-1.305598199544785) q[7];
rz(-1.736015882308675) q[7];
ry(2.247619853723386) q[8];
rz(0.7988403138718994) q[8];
ry(-1.4906317542021135) q[9];
rz(1.8452621343557043) q[9];
ry(1.919492381328661) q[10];
rz(-0.704625036454642) q[10];
ry(0.32093809088983427) q[11];
rz(-2.4593027614126073) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.8551191855565072) q[0];
rz(0.7474389360465743) q[0];
ry(-2.9752796923492744) q[1];
rz(1.5938200541006005) q[1];
ry(-0.8895417828982464) q[2];
rz(-2.5741771240155598) q[2];
ry(1.2930578614450283) q[3];
rz(-0.9924658982607442) q[3];
ry(-0.4030127366340056) q[4];
rz(-3.0361157704237063) q[4];
ry(1.9139851385556994) q[5];
rz(1.0466397962604344) q[5];
ry(-0.8237904417177324) q[6];
rz(-0.1362178402723737) q[6];
ry(0.4816198605700049) q[7];
rz(-1.2559459617473687) q[7];
ry(-2.6832736748991395) q[8];
rz(-1.0938637892383722) q[8];
ry(-1.3251442431421276) q[9];
rz(-2.637107064888545) q[9];
ry(1.3870956128845593) q[10];
rz(1.187339816252514) q[10];
ry(0.47184122098375136) q[11];
rz(0.8544840969019996) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.6976445494576793) q[0];
rz(-2.4537987078937595) q[0];
ry(0.3633319388958025) q[1];
rz(0.7906112592145185) q[1];
ry(0.020842775795973677) q[2];
rz(2.885638429097486) q[2];
ry(2.030821953909566) q[3];
rz(-2.4146811275455633) q[3];
ry(-2.1607831141382494) q[4];
rz(-0.3978687585421667) q[4];
ry(-0.5085502975968074) q[5];
rz(-1.4295786065759444) q[5];
ry(-1.6727040549705148) q[6];
rz(-2.6102838172321308) q[6];
ry(2.0287412885123164) q[7];
rz(-1.144329914071451) q[7];
ry(-2.5441177000573205) q[8];
rz(-1.9160109700058943) q[8];
ry(2.8848059699479025) q[9];
rz(1.5392582177926295) q[9];
ry(-2.441780361650502) q[10];
rz(0.8621568513585717) q[10];
ry(1.9717109812185472) q[11];
rz(0.35685952083977485) q[11];