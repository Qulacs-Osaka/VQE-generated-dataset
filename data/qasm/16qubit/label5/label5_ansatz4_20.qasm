OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.6044236081143701) q[0];
rz(-0.014517240856323622) q[0];
ry(1.1331983052868004) q[1];
rz(0.7782844541416064) q[1];
ry(-1.5950786076772812) q[2];
rz(2.1195992660261815) q[2];
ry(-2.0565330400595707) q[3];
rz(1.9249842001068815) q[3];
ry(1.6342069985309846) q[4];
rz(1.573071577687795) q[4];
ry(-1.6405350695405325) q[5];
rz(-1.5814967528038029) q[5];
ry(-1.6029679499095764) q[6];
rz(1.571121160927023) q[6];
ry(0.0012022861140952301) q[7];
rz(0.35878772648902146) q[7];
ry(3.141106282558874) q[8];
rz(-0.3528634848928878) q[8];
ry(-0.002563970273413041) q[9];
rz(-0.10155745825075553) q[9];
ry(2.3837682957834683) q[10];
rz(-1.6125647397933573) q[10];
ry(2.712115096456322) q[11];
rz(-1.4149854525313792) q[11];
ry(3.1332623441537892) q[12];
rz(2.941477670333204) q[12];
ry(-0.0065079339019424864) q[13];
rz(2.3611294260372393) q[13];
ry(1.865359680167055) q[14];
rz(-2.338229214516405) q[14];
ry(1.5424485514774584) q[15];
rz(0.6681793098097213) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.1710033122700345) q[0];
rz(0.9326525140466675) q[0];
ry(2.8892953411922595) q[1];
rz(2.2447918705814396) q[1];
ry(-0.011606246411848707) q[2];
rz(-0.44665191500588297) q[2];
ry(0.07537596066110996) q[3];
rz(-2.613827904538653) q[3];
ry(-1.5698920710932818) q[4];
rz(-3.058747710992681) q[4];
ry(-1.5711770580294016) q[5];
rz(-0.08179692009988386) q[5];
ry(2.9911720568552735) q[6];
rz(-2.9314315006266596) q[6];
ry(3.141517737739677) q[7];
rz(-1.2827949566984194) q[7];
ry(1.5720370008850022) q[8];
rz(-0.1444129562604478) q[8];
ry(-1.6023891323892574) q[9];
rz(3.140833103156173) q[9];
ry(1.5903089370021872) q[10];
rz(0.6166379236144879) q[10];
ry(1.5396028518594898) q[11];
rz(1.9936766404235753) q[11];
ry(-0.0018609783998888574) q[12];
rz(-0.4505905900133893) q[12];
ry(-0.00012929904654815372) q[13];
rz(0.5450218950069923) q[13];
ry(-0.02392686671630706) q[14];
rz(0.4153909058658529) q[14];
ry(-0.008579957714248904) q[15];
rz(0.7268749745085533) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.01036966019722) q[0];
rz(0.2713498719241899) q[0];
ry(-1.1694159210071389) q[1];
rz(0.17101760413394107) q[1];
ry(1.5695596203785052) q[2];
rz(-3.07905418193266) q[2];
ry(2.9508532217289343) q[3];
rz(1.479115138130244) q[3];
ry(-1.5650021574281439) q[4];
rz(1.642693761854881) q[4];
ry(-1.5658436824036475) q[5];
rz(1.532405816348113) q[5];
ry(0.00030293541660250645) q[6];
rz(-1.0337027009277238) q[6];
ry(-3.1414677839310783) q[7];
rz(0.8999188624206489) q[7];
ry(-3.1219065682988907) q[8];
rz(-1.7181757540334102) q[8];
ry(-1.536095936170991) q[9];
rz(1.7151338886206728) q[9];
ry(0.00666972732729576) q[10];
rz(-1.6581475314491645) q[10];
ry(-0.007944955209090061) q[11];
rz(-3.0374281459953343) q[11];
ry(1.1763986579526198) q[12];
rz(-0.36796994420458173) q[12];
ry(0.008716133521607583) q[13];
rz(2.3135235619225942) q[13];
ry(-2.4499202677286687) q[14];
rz(2.7900072580052147) q[14];
ry(0.42843793215316806) q[15];
rz(-0.5250970929881839) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5357491796614913) q[0];
rz(0.3312283738094588) q[0];
ry(-1.361562434617504) q[1];
rz(1.888811405651095) q[1];
ry(0.42066463912823426) q[2];
rz(-0.8810560302827234) q[2];
ry(1.5142601572419945) q[3];
rz(-3.03967899726472) q[3];
ry(1.538209054491848) q[4];
rz(2.3667310156250623) q[4];
ry(-1.5738638256729134) q[5];
rz(-1.227115338134075) q[5];
ry(-1.5977671789147863) q[6];
rz(2.6425179915453327) q[6];
ry(3.123635477073875) q[7];
rz(-2.8904857350155506) q[7];
ry(-1.774590221618416) q[8];
rz(-0.09160750641941551) q[8];
ry(0.22598790370801589) q[9];
rz(1.4171625145151874) q[9];
ry(2.3788339688396807) q[10];
rz(3.008059226535759) q[10];
ry(2.3211325306522084) q[11];
rz(0.11493058755909936) q[11];
ry(-0.0028784216728124272) q[12];
rz(2.369044006044817) q[12];
ry(0.002493355120347368) q[13];
rz(1.124141566669057) q[13];
ry(-3.1304276171598064) q[14];
rz(-0.42615999388480014) q[14];
ry(-3.138894205018399) q[15];
rz(-1.4380811118589227) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.494335258656104) q[0];
rz(-2.266238027109593) q[0];
ry(0.8779815485721666) q[1];
rz(-2.1408348280441687) q[1];
ry(-0.905758820857637) q[2];
rz(-1.6629836563495963) q[2];
ry(3.104766951377681) q[3];
rz(-0.9080006535505296) q[3];
ry(3.1415523066178954) q[4];
rz(-0.5552532882253711) q[4];
ry(0.0005701700088183159) q[5];
rz(2.60840070773942) q[5];
ry(-0.13496376587376901) q[6];
rz(2.432913752708398) q[6];
ry(0.0036404931402024827) q[7];
rz(0.9145220408257159) q[7];
ry(-0.01572829434487133) q[8];
rz(-3.104839147094264) q[8];
ry(0.015184718406502916) q[9];
rz(-2.588139341655503) q[9];
ry(-1.5704255990671523) q[10];
rz(2.0988813065283147) q[10];
ry(1.5964667744413106) q[11];
rz(-1.5299465599653344) q[11];
ry(3.0901868670391015) q[12];
rz(3.0922818975177164) q[12];
ry(-0.028001638116191607) q[13];
rz(2.1956414077912996) q[13];
ry(-3.0992437065318965) q[14];
rz(1.016255172634156) q[14];
ry(-0.08909480784022782) q[15];
rz(0.03665869148563376) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.2604570281414897) q[0];
rz(3.034838228594196) q[0];
ry(-2.1748476839519872) q[1];
rz(-0.3395895635762294) q[1];
ry(-1.389063173006005) q[2];
rz(0.32324092381469716) q[2];
ry(-2.8429813865416116) q[3];
rz(-0.43197878031455866) q[3];
ry(1.580661670042189) q[4];
rz(0.7466348913305979) q[4];
ry(1.5643515359343918) q[5];
rz(1.4475450172272897) q[5];
ry(0.20471822171995321) q[6];
rz(-1.2144644946491578) q[6];
ry(0.01900401902994637) q[7];
rz(2.502136389351744) q[7];
ry(-0.6459122747423204) q[8];
rz(1.6979174344645953) q[8];
ry(0.012942023382702759) q[9];
rz(-0.5687622937627932) q[9];
ry(0.052614286683128925) q[10];
rz(2.194389268236755) q[10];
ry(-0.004169919250773384) q[11];
rz(-1.3080206381506922) q[11];
ry(3.113323372942944) q[12];
rz(-1.5532666559932462) q[12];
ry(1.4189004084904646) q[13];
rz(-1.2208170588549763) q[13];
ry(1.408003995707939) q[14];
rz(1.5784255335013755) q[14];
ry(2.9395659360791435) q[15];
rz(2.427817261371296) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.8670624877379725) q[0];
rz(0.6678744583613389) q[0];
ry(2.1108277737172063) q[1];
rz(1.5654516248842612) q[1];
ry(-2.155249255239966) q[2];
rz(-0.3555983919357147) q[2];
ry(-0.7064830004495928) q[3];
rz(1.28288728533277) q[3];
ry(-0.0008660687320192148) q[4];
rz(-0.7442175564450556) q[4];
ry(0.00342672113297242) q[5];
rz(-1.4282811554387447) q[5];
ry(0.01602724626126172) q[6];
rz(-0.6728877721386581) q[6];
ry(-0.0015859937666178872) q[7];
rz(-0.6451495123237204) q[7];
ry(0.1685376331644702) q[8];
rz(2.562288466813135) q[8];
ry(3.081047791901946) q[9];
rz(-1.1319184939590938) q[9];
ry(2.376555620742238e-06) q[10];
rz(1.6237040986549545) q[10];
ry(-3.139413442445336) q[11];
rz(2.480913593476605) q[11];
ry(-3.1363028756697324) q[12];
rz(-2.6848977417033266) q[12];
ry(-0.08907424177511114) q[13];
rz(2.6433750380522127) q[13];
ry(2.666647326132397) q[14];
rz(2.3143350794708373) q[14];
ry(-0.46926806997192916) q[15];
rz(2.8010630217923436) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.7446678931410062) q[0];
rz(2.3145500737409717) q[0];
ry(0.9189244506651493) q[1];
rz(-2.818096466762624) q[1];
ry(1.1886802966861412) q[2];
rz(-1.979860486311665) q[2];
ry(-1.1462994479972592) q[3];
rz(-2.949558348096072) q[3];
ry(-1.5069332111423177) q[4];
rz(2.0417443647732263) q[4];
ry(-1.6628212342907815) q[5];
rz(-1.0760550284567278) q[5];
ry(1.4376284449348518) q[6];
rz(0.5571997792389324) q[6];
ry(-1.5584825863559582) q[7];
rz(-0.018019894045272267) q[7];
ry(-3.1028009565565275) q[8];
rz(-2.6448453815370123) q[8];
ry(-3.1155892650048007) q[9];
rz(-0.9212635942855103) q[9];
ry(0.04227767876972167) q[10];
rz(-0.2399281051618447) q[10];
ry(-3.059172803452672) q[11];
rz(-0.29858390911342) q[11];
ry(1.5788655480895495) q[12];
rz(-0.3464104935322966) q[12];
ry(-0.9615370432739101) q[13];
rz(-1.3706981175340152) q[13];
ry(-3.1104431061304414) q[14];
rz(-0.02292137736408684) q[14];
ry(-2.962580365324572) q[15];
rz(0.10629909291086559) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.31872729783035636) q[0];
rz(1.1806917210972019) q[0];
ry(-2.182565958785598) q[1];
rz(-2.1743803986916115) q[1];
ry(1.0548678825887632) q[2];
rz(-0.275822735425602) q[2];
ry(0.45568522200291006) q[3];
rz(3.0025971747655005) q[3];
ry(-0.08264245037831557) q[4];
rz(-2.6333928288229136) q[4];
ry(3.028462209212513) q[5];
rz(1.3486875314278768) q[5];
ry(1.5731035340435868) q[6];
rz(-1.2600858314874235) q[6];
ry(-1.5704948938109613) q[7];
rz(-0.07025142117836078) q[7];
ry(3.14144313705716) q[8];
rz(-0.5762797543135298) q[8];
ry(3.1410843271750175) q[9];
rz(0.19121632997714136) q[9];
ry(1.5762234640238475) q[10];
rz(-2.9615802142683583) q[10];
ry(1.566385568918749) q[11];
rz(0.12787550153790628) q[11];
ry(-0.1555565024948032) q[12];
rz(0.4966024030138423) q[12];
ry(3.085387499696401) q[13];
rz(1.1371765490582024) q[13];
ry(-1.9836634552992118) q[14];
rz(1.5200930276669276) q[14];
ry(1.1310955952575603) q[15];
rz(-1.1712762124506535) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.7412770541539384) q[0];
rz(2.9712837553592015) q[0];
ry(2.4042547917782215) q[1];
rz(2.993430976413821) q[1];
ry(-1.5458877284759576) q[2];
rz(-2.44348666500561) q[2];
ry(3.133776144990788) q[3];
rz(-1.5547264728991765) q[3];
ry(3.124797147614172) q[4];
rz(2.8106488612197946) q[4];
ry(-3.1164256196178983) q[5];
rz(-0.6033454960988177) q[5];
ry(0.03457257255289826) q[6];
rz(-0.5944008072460579) q[6];
ry(1.5833298124166948) q[7];
rz(2.812403886484037) q[7];
ry(-0.496356849149727) q[8];
rz(0.3525104962999608) q[8];
ry(1.5713346568620994) q[9];
rz(-1.4367494553705464) q[9];
ry(2.8484245574862914) q[10];
rz(-2.823001927143014) q[10];
ry(-0.7298620538843839) q[11];
rz(-0.17037628566305438) q[11];
ry(3.1300792848139283) q[12];
rz(-2.892535297441219) q[12];
ry(-3.1377014027615657) q[13];
rz(0.23596064030867842) q[13];
ry(-3.1360811818744123) q[14];
rz(0.9682178288286369) q[14];
ry(-0.0048467846559572675) q[15];
rz(-0.5126839529217784) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.5920014909345683) q[0];
rz(2.540645283752028) q[0];
ry(2.563442276451453) q[1];
rz(0.5984665979549612) q[1];
ry(-2.2139657940721174) q[2];
rz(-2.768581274208667) q[2];
ry(-1.5100386655672853) q[3];
rz(-2.085639935264199) q[3];
ry(0.07881707940866221) q[4];
rz(-0.21955107431747936) q[4];
ry(-2.7922341918473377) q[5];
rz(3.108107895815424) q[5];
ry(-0.0028729204792383493) q[6];
rz(-2.873689012252078) q[6];
ry(-0.0003481529263687609) q[7];
rz(-0.5565816996332771) q[7];
ry(3.139779311548785) q[8];
rz(-2.7962637545077746) q[8];
ry(-3.1409700000585445) q[9];
rz(2.767276414444873) q[9];
ry(-1.5711863130395791) q[10];
rz(2.4609357755498946) q[10];
ry(-1.5662530993585895) q[11];
rz(-2.0633952137818747) q[11];
ry(-2.9859315592877502) q[12];
rz(-2.0575702687440405) q[12];
ry(-1.6062866326889451) q[13];
rz(-0.13702274778582524) q[13];
ry(-2.752450933576712) q[14];
rz(-0.6235924698735538) q[14];
ry(0.8800550321232397) q[15];
rz(-0.358519187224467) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.938037258691895) q[0];
rz(2.970731517913514) q[0];
ry(-1.9048276622894824) q[1];
rz(0.07530155722613507) q[1];
ry(1.5557677905788936) q[2];
rz(1.923805392234911) q[2];
ry(-1.529007827823678) q[3];
rz(1.569274052461145) q[3];
ry(-1.6136449196617577) q[4];
rz(2.296223853546561) q[4];
ry(-1.556324619797439) q[5];
rz(-3.1173939329015052) q[5];
ry(0.2754905355317418) q[6];
rz(0.45099200643150844) q[6];
ry(-0.003561805351988873) q[7];
rz(-0.4247051385606992) q[7];
ry(-3.136810441656868) q[8];
rz(0.7978842487498543) q[8];
ry(-1.5704265602368395) q[9];
rz(-2.8007723825102557) q[9];
ry(-3.113310023029379) q[10];
rz(-0.6183483214271384) q[10];
ry(-3.004092905338766) q[11];
rz(1.2289496905605901) q[11];
ry(0.006436165153925089) q[12];
rz(0.7915327722515049) q[12];
ry(-3.1010097137681814) q[13];
rz(2.485241132592998) q[13];
ry(-2.3376759396578892) q[14];
rz(0.11949270919203236) q[14];
ry(1.1495611770434646) q[15];
rz(1.6928785710728893) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.0288738044284593) q[0];
rz(1.3632693604389257) q[0];
ry(1.42571815627077) q[1];
rz(1.4660760367782903) q[1];
ry(1.5298586694169147) q[2];
rz(1.4202749424790504) q[2];
ry(1.5527912954597354) q[3];
rz(0.05167322630363091) q[3];
ry(-1.3080351754793886) q[4];
rz(1.5582940432995978) q[4];
ry(1.2083577505912144) q[5];
rz(1.533839119756684) q[5];
ry(0.002366279610966693) q[6];
rz(-0.17063509444401193) q[6];
ry(0.0007940245726034689) q[7];
rz(-0.1868129226322113) q[7];
ry(-0.0019425845750559298) q[8];
rz(-0.9000357370675295) q[8];
ry(-0.00016757993912897733) q[9];
rz(-1.4934114298945342) q[9];
ry(2.3447298334439717) q[10];
rz(0.02913941838889134) q[10];
ry(1.7210953345653837) q[11];
rz(-3.116418392148084) q[11];
ry(3.128027749257054) q[12];
rz(1.543243139039715) q[12];
ry(0.015119846825886052) q[13];
rz(-2.7750217590025503) q[13];
ry(-2.869812464147417) q[14];
rz(1.6615922654979993) q[14];
ry(1.8585006301319105) q[15];
rz(-1.0193468374419616) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.3492364024611465) q[0];
rz(1.5369872819038743) q[0];
ry(-1.3774260063825563) q[1];
rz(1.5173493017779816) q[1];
ry(0.09125267566492747) q[2];
rz(1.681037554963444) q[2];
ry(-1.6070978230805897) q[3];
rz(1.5449031831041298) q[3];
ry(3.121090245006107) q[4];
rz(2.582687579655484) q[4];
ry(2.9177139173471143) q[5];
rz(-0.08091490122471434) q[5];
ry(0.05258479011500513) q[6];
rz(-1.1450534115322677) q[6];
ry(1.184698967280401) q[7];
rz(-0.8891288083625299) q[7];
ry(-0.0411075935615024) q[8];
rz(1.541456695556644) q[8];
ry(-3.1393877240332033) q[9];
rz(2.404662839261449) q[9];
ry(1.5556590577919032) q[10];
rz(1.48612164131316) q[10];
ry(-1.5318939146697816) q[11];
rz(0.9699776768579083) q[11];
ry(-3.1286139717550143) q[12];
rz(0.7918125445968016) q[12];
ry(-0.011804613302614024) q[13];
rz(0.3884095476981418) q[13];
ry(-2.141491750379978) q[14];
rz(0.463601407356518) q[14];
ry(1.952134042429347) q[15];
rz(-2.7565712292200253) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5688987112455326) q[0];
rz(-1.58801587031959) q[0];
ry(-1.5719928847517375) q[1];
rz(1.0700822348812145) q[1];
ry(-1.558732556739053) q[2];
rz(0.025486001665712802) q[2];
ry(-1.6644913188514556) q[3];
rz(-1.621624016866384) q[3];
ry(-3.0401561235284267) q[4];
rz(0.3505643922812407) q[4];
ry(0.10494017540281408) q[5];
rz(0.31972299287531863) q[5];
ry(0.0028924819210569223) q[6];
rz(-0.28215063761598075) q[6];
ry(-3.141516690406186) q[7];
rz(-0.7214875681663546) q[7];
ry(-0.0005048745949681984) q[8];
rz(-1.471484530040744) q[8];
ry(3.140152377499521) q[9];
rz(-1.4328632718247696) q[9];
ry(1.7371931326693841) q[10];
rz(-2.990707165539124) q[10];
ry(-0.03540823023973694) q[11];
rz(-2.000831636780887) q[11];
ry(-0.00036148407907899755) q[12];
rz(-2.611939990527517) q[12];
ry(3.131110963918572) q[13];
rz(0.06482692765359697) q[13];
ry(-0.252319164918517) q[14];
rz(-0.7631250774642124) q[14];
ry(2.6670501488123652) q[15];
rz(-2.876782519189109) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.04493561133306567) q[0];
rz(-2.7156951853407016) q[0];
ry(-0.0028926734745823106) q[1];
rz(0.5268032006339274) q[1];
ry(1.6022908625361083) q[2];
rz(3.0934298365515662) q[2];
ry(-1.4759951918929073) q[3];
rz(-2.914965718027872) q[3];
ry(-0.027513812299641307) q[4];
rz(2.9532641026519286) q[4];
ry(3.1364633126788064) q[5];
rz(-0.7009086882278981) q[5];
ry(0.02949566359900313) q[6];
rz(1.9159219903042493) q[6];
ry(-2.360241233792569) q[7];
rz(2.1912656691061185) q[7];
ry(-1.4837343541579378) q[8];
rz(1.0739684466943242) q[8];
ry(2.839946753694843) q[9];
rz(3.0066400445871855) q[9];
ry(-0.0020700691694530704) q[10];
rz(-1.8168995451200094) q[10];
ry(-2.9627099189660124) q[11];
rz(1.4812968720845185) q[11];
ry(-1.5741006148116958) q[12];
rz(-2.076737777140755) q[12];
ry(-3.140395592265824) q[13];
rz(-1.7032555413226107) q[13];
ry(-2.3722180881235007) q[14];
rz(1.812649695155585) q[14];
ry(-2.853534178982494) q[15];
rz(2.3298371783588747) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.4625153905887576) q[0];
rz(2.124182491251731) q[0];
ry(0.007310926781433395) q[1];
rz(0.4373319325875846) q[1];
ry(-1.5773966277625577) q[2];
rz(1.2314343582073008) q[2];
ry(3.120613220951859) q[3];
rz(1.5713160553660428) q[3];
ry(3.1140119343443353) q[4];
rz(0.47171314408420084) q[4];
ry(-0.07876428575274375) q[5];
rz(0.9646552326217029) q[5];
ry(-3.1411316053212763) q[6];
rz(0.12931104118558512) q[6];
ry(6.464424009525933e-06) q[7];
rz(0.9567982881127096) q[7];
ry(3.1412680454943342) q[8];
rz(0.87705732541849) q[8];
ry(-3.1408983366382057) q[9];
rz(1.6995796810753907) q[9];
ry(0.0020139887099555054) q[10];
rz(2.8499501741697) q[10];
ry(3.1408858064918213) q[11];
rz(-2.2514033244859313) q[11];
ry(0.0006996511028354677) q[12];
rz(-2.6347861948284246) q[12];
ry(-1.5749210089780243) q[13];
rz(1.5710761205759685) q[13];
ry(-1.5597702085455836) q[14];
rz(-3.138155733297516) q[14];
ry(1.573016855518734) q[15];
rz(-1.57142370155335) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.1385751018462265) q[0];
rz(-2.8344690879436008) q[0];
ry(-0.008699186754722586) q[1];
rz(-0.45783887073383944) q[1];
ry(3.137844128834464) q[2];
rz(-2.61978925723014) q[2];
ry(-3.140360778278297) q[3];
rz(-1.7885610220227892) q[3];
ry(2.6399257910855454) q[4];
rz(2.0688834307987705) q[4];
ry(-1.6021726119309898) q[5];
rz(-0.05286985603376071) q[5];
ry(3.069443908929093) q[6];
rz(-0.9739011987681394) q[6];
ry(-2.755221347347559) q[7];
rz(-1.1816007390657512) q[7];
ry(0.09598286823280404) q[8];
rz(1.1168885035753136) q[8];
ry(1.4821089099295737) q[9];
rz(0.3202313902121831) q[9];
ry(0.001454487117882533) q[10];
rz(-1.1489113080562883) q[10];
ry(-3.140892835592077) q[11];
rz(-1.666609230339846) q[11];
ry(1.5705195045477893) q[12];
rz(1.6818124381240267) q[12];
ry(-1.5686988487053366) q[13];
rz(2.8920378423791653) q[13];
ry(1.577062952560561) q[14];
rz(3.1407176463322153) q[14];
ry(-1.5729302174087765) q[15];
rz(0.6901221263948577) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.41270657411596373) q[0];
rz(1.8855730280595664) q[0];
ry(-1.5626114366937816) q[1];
rz(0.04462920184807829) q[1];
ry(0.024084100420727953) q[2];
rz(0.6235847702094882) q[2];
ry(-1.5647230165704968) q[3];
rz(-0.05255526853610649) q[3];
ry(0.010991300115831493) q[4];
rz(1.6582021799717817) q[4];
ry(-1.6073013755696852) q[5];
rz(0.3809819253683448) q[5];
ry(-3.14153200104533) q[6];
rz(-1.9237792263089364) q[6];
ry(-0.001407970819422566) q[7];
rz(0.7660450243703574) q[7];
ry(3.140035673235862) q[8];
rz(-2.4556647366810225) q[8];
ry(3.14070138529135) q[9];
rz(0.07406409259935742) q[9];
ry(-1.6290403324151415) q[10];
rz(1.8194453518334157) q[10];
ry(1.5656683966023313) q[11];
rz(1.202788801331808) q[11];
ry(-1.7127312762151923) q[12];
rz(2.2972313923727525) q[12];
ry(0.000834256446705764) q[13];
rz(-2.8832101069930047) q[13];
ry(1.5710892415162248) q[14];
rz(-1.5020369766555817) q[14];
ry(-0.0004584662729048005) q[15];
rz(-2.996213709779677) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.7500965269192863) q[0];
rz(0.0023066313350508898) q[0];
ry(-1.562512659838341) q[1];
rz(0.13489659508193697) q[1];
ry(1.5713658259770051) q[2];
rz(0.45371927622040165) q[2];
ry(3.1397988846743354) q[3];
rz(-0.25462511798338644) q[3];
ry(-0.0305187219330616) q[4];
rz(-2.181322869311944) q[4];
ry(-0.023847973749796303) q[5];
rz(-1.72031396103724) q[5];
ry(1.5619432218514429) q[6];
rz(1.4698738938619476) q[6];
ry(0.009488603058486866) q[7];
rz(-2.6274988342609467) q[7];
ry(1.542302986882417) q[8];
rz(-2.805685022312644) q[8];
ry(1.5671976075869143) q[9];
rz(2.8041426085788546) q[9];
ry(3.141423554956415) q[10];
rz(1.9591545701297695) q[10];
ry(8.260120257691668e-06) q[11];
rz(-0.9454467523787133) q[11];
ry(1.5700851492032035) q[12];
rz(-1.0810254017585927) q[12];
ry(1.5707325157002847) q[13];
rz(2.027059906666249) q[13];
ry(3.005981531712401) q[14];
rz(0.09187264536877038) q[14];
ry(-0.03208871324464635) q[15];
rz(-0.8743790979973226) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5822832964393427) q[0];
rz(-3.0326182233098247) q[0];
ry(-0.017265925916387204) q[1];
rz(0.6425961136984695) q[1];
ry(-0.02065963644762168) q[2];
rz(-2.630366499364731) q[2];
ry(3.126997667013081) q[3];
rz(2.927380505540163) q[3];
ry(0.42816208450474924) q[4];
rz(0.6035840309332717) q[4];
ry(0.12238979253748446) q[5];
rz(-0.17640537099804202) q[5];
ry(-7.882725523902976e-05) q[6];
rz(-2.853788734160558) q[6];
ry(0.00036706325975278937) q[7];
rz(2.86899458630718) q[7];
ry(8.835943455373656e-06) q[8];
rz(1.2402647738319992) q[8];
ry(-0.00033260608752172516) q[9];
rz(1.9777033660432588) q[9];
ry(-3.1270651846262676) q[10];
rz(-3.102972992705155) q[10];
ry(-0.005432547313843913) q[11];
rz(1.313701383085208) q[11];
ry(-0.0013483503633062953) q[12];
rz(-2.0596979798472583) q[12];
ry(1.6907151805033527) q[13];
rz(-1.5093369094231082) q[13];
ry(-1.5815581093959183) q[14];
rz(-2.4175867764146326) q[14];
ry(-1.580546390183892) q[15];
rz(-1.4187963765699008) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.689951707630519) q[0];
rz(1.6639183600666385) q[0];
ry(0.002043141532801449) q[1];
rz(0.6720137036667156) q[1];
ry(3.1409772897756265) q[2];
rz(1.1658592901069957) q[2];
ry(-1.5776914381041554) q[3];
rz(-2.2094722959600546) q[3];
ry(-0.5215083512864735) q[4];
rz(-2.4206965926488397) q[4];
ry(1.5045561333890163) q[5];
rz(2.1671984663790704) q[5];
ry(0.3795686396807483) q[6];
rz(-2.0420834059848008) q[6];
ry(0.12800346622350234) q[7];
rz(0.2496381756019499) q[7];
ry(1.8074988661683955) q[8];
rz(2.720830484253557) q[8];
ry(-0.04626114361923218) q[9];
rz(-0.08210158076011426) q[9];
ry(1.5701152557343248) q[10];
rz(-0.00098805004231739) q[10];
ry(1.5717142658289234) q[11];
rz(-0.0007462176211250338) q[11];
ry(1.5221078664765215) q[12];
rz(-1.712434083997917) q[12];
ry(-1.569181465678783) q[13];
rz(1.5993504053165184) q[13];
ry(-1.5709953500753189) q[14];
rz(-0.002035561045931722) q[14];
ry(3.088619434822126) q[15];
rz(1.401942524025622) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5663157044886216) q[0];
rz(3.0605755978107654) q[0];
ry(-3.0104624187398) q[1];
rz(1.5993852451657387) q[1];
ry(0.000611619376162603) q[2];
rz(2.1161071320503337) q[2];
ry(3.1382984190289025) q[3];
rz(-0.6380201954866855) q[3];
ry(3.1146147382489273) q[4];
rz(-0.3861700881250005) q[4];
ry(-0.0013384194958518947) q[5];
rz(0.36535763467749843) q[5];
ry(-0.00030781522354761387) q[6];
rz(-2.850785138234443) q[6];
ry(-0.0025432193785785984) q[7];
rz(1.3480416021990376) q[7];
ry(0.0008672313249364904) q[8];
rz(-1.155387121961322) q[8];
ry(2.6581754254628717e-05) q[9];
rz(2.2103798350772275) q[9];
ry(1.5715530553823982) q[10];
rz(-1.1685218888865583) q[10];
ry(1.5708773781927505) q[11];
rz(3.1365788477401395) q[11];
ry(0.0014610325039141614) q[12];
rz(0.15324393145463058) q[12];
ry(0.02376377679171244) q[13];
rz(1.5427933967769558) q[13];
ry(-1.5725607636470693) q[14];
rz(-1.5657956500568682) q[14];
ry(-1.5717454884799942) q[15];
rz(1.5558941387801228) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.0467876746331397) q[0];
rz(0.3674543779218256) q[0];
ry(-0.11126871405199912) q[1];
rz(1.5424591964637218) q[1];
ry(0.0015701744736515977) q[2];
rz(2.7016384522387784) q[2];
ry(1.5715006356969778) q[3];
rz(-3.023622940427663) q[3];
ry(1.7916917169802182) q[4];
rz(2.9657384230099226) q[4];
ry(-0.08745897076959883) q[5];
rz(0.7346238984740164) q[5];
ry(-1.5081869526632972) q[6];
rz(-1.6376230746439837) q[6];
ry(1.5692037250511888) q[7];
rz(-1.2841642901730772) q[7];
ry(-3.0778155832478657) q[8];
rz(-1.2957563970947572) q[8];
ry(-0.0017425580521820484) q[9];
rz(1.088094347704148) q[9];
ry(-0.0002851231295412084) q[10];
rz(-0.09079233145727647) q[10];
ry(-1.568546518089252) q[11];
rz(0.24620200566405706) q[11];
ry(-1.5728330676012208) q[12];
rz(0.30946730307256165) q[12];
ry(-1.568504147467365) q[13];
rz(0.14554714487678222) q[13];
ry(1.5209666886870932) q[14];
rz(0.31052548822292775) q[14];
ry(-0.48888932386766193) q[15];
rz(-1.4153269608540313) q[15];