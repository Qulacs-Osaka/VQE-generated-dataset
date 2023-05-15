OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(3.1415801194019153) q[0];
rz(1.1273547625021276) q[0];
ry(1.5409501086082136) q[1];
rz(-0.2676151593353718) q[1];
ry(-3.14154784387757) q[2];
rz(-3.1086494609365567) q[2];
ry(-0.25478850643709056) q[3];
rz(-1.6326778959170483) q[3];
ry(-0.1313817006777428) q[4];
rz(-3.1386436921284684) q[4];
ry(1.5707721664990153) q[5];
rz(-0.0014185068220918294) q[5];
ry(-1.5709260186084928) q[6];
rz(2.166735573040003) q[6];
ry(1.5707935823747574) q[7];
rz(-1.5707515753905552) q[7];
ry(-3.14159237753082) q[8];
rz(-2.8449521321148397) q[8];
ry(-1.0684053623144507e-05) q[9];
rz(2.206492648343543) q[9];
ry(-1.5325062986531446) q[10];
rz(-1.308543758781064) q[10];
ry(2.4942575849222337e-06) q[11];
rz(0.6677359184073186) q[11];
ry(-2.0757904899781008e-07) q[12];
rz(0.01614231318722226) q[12];
ry(1.862571519239964e-07) q[13];
rz(-1.390454183040334) q[13];
ry(1.570805565897551) q[14];
rz(2.076849965098333) q[14];
ry(-1.570717870145892) q[15];
rz(-3.1415925779237717) q[15];
ry(-1.5707958602522023) q[16];
rz(1.5704660946551892) q[16];
ry(-5.290423965575997e-07) q[17];
rz(-0.13669012678919026) q[17];
ry(-3.1312394891969593) q[18];
rz(-1.5695061784882987) q[18];
ry(-3.1415924567471922) q[19];
rz(1.5834251184002053) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5708088243105154) q[0];
rz(0.010360789899780622) q[0];
ry(-0.2254214733994932) q[1];
rz(1.4397621739407283) q[1];
ry(-1.5721051696295731) q[2];
rz(1.5708054268853775) q[2];
ry(1.5707966424537831) q[3];
rz(1.5992195950319135) q[3];
ry(0.48272827419833386) q[4];
rz(-1.57120224444735) q[4];
ry(3.1166444183608104) q[5];
rz(1.5693490213425827) q[5];
ry(0.13795290718754494) q[6];
rz(2.377407488051599) q[6];
ry(-1.5707490713309404) q[7];
rz(-2.630140027883629) q[7];
ry(2.6182993133772222e-08) q[8];
rz(-1.6557989678896645) q[8];
ry(7.062198258189768e-06) q[9];
rz(-2.7506543253585116) q[9];
ry(-2.9948959172360405) q[10];
rz(0.2649699308593032) q[10];
ry(-0.00018097130866578937) q[11];
rz(-2.6542230071627535) q[11];
ry(-1.570796260964996) q[12];
rz(1.5707968745677734) q[12];
ry(1.570795556058282) q[13];
rz(1.5707989620672518) q[13];
ry(5.98663794093568e-08) q[14];
rz(2.6355587754993817) q[14];
ry(-1.5804883268183385) q[15];
rz(3.1415916459505713) q[15];
ry(-1.5388957520529483) q[16];
rz(-1.5611046709927077) q[16];
ry(-0.00019857149316049316) q[17];
rz(1.7735672189660816) q[17];
ry(1.5708070673305121) q[18];
rz(1.5707929074909648) q[18];
ry(-1.5708147485903847) q[19];
rz(-1.5432254796730147) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.0389639429790894) q[0];
rz(3.1311429705701457) q[0];
ry(-0.0032555095937096897) q[1];
rz(1.9778098340237251) q[1];
ry(-1.5708070294127499) q[2];
rz(0.02309765970777277) q[2];
ry(0.039907170775010416) q[3];
rz(1.5032157429482116) q[3];
ry(-2.4207710694520035) q[4];
rz(3.136096334066714) q[4];
ry(-0.12035388709649646) q[5];
rz(1.2377239742857578) q[5];
ry(1.570795711321207) q[6];
rz(-2.286209402696973) q[6];
ry(1.5690551303643006) q[7];
rz(-0.0005997031391800078) q[7];
ry(0.0016691020651390684) q[8];
rz(-2.4721043312703492) q[8];
ry(3.141592603993819) q[9];
rz(1.3954375268039572) q[9];
ry(1.5708041349812003) q[10];
rz(0.21977870738116484) q[10];
ry(-1.5707977443724432) q[11];
rz(3.1415922561087317) q[11];
ry(1.5707958111282916) q[12];
rz(1.2030475079974834) q[12];
ry(-1.570794742014229) q[13];
rz(1.9385412681499052) q[13];
ry(-1.4159374248561578) q[14];
rz(-7.83025217643285e-06) q[14];
ry(1.5707962012589136) q[15];
rz(0.6441664524701662) q[15];
ry(-1.57079492726825) q[16];
rz(0.7167919782953263) q[16];
ry(-1.5707962313959545) q[17];
rz(1.5707957920879059) q[17];
ry(-1.5708016314055446) q[18];
rz(-3.1121047575454788) q[18];
ry(2.3985650922200263) q[19];
rz(0.037425045534568824) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.0121305651990333) q[0];
rz(-1.5709463043523382) q[0];
ry(-1.5708054993275185) q[1];
rz(0.005085187084043596) q[1];
ry(3.1411648249392767) q[2];
rz(1.5949956642403365) q[2];
ry(0.0053679175735048545) q[3];
rz(0.03913322313062295) q[3];
ry(3.112982806347207) q[4];
rz(-1.57567155955401) q[4];
ry(0.0001616714976158185) q[5];
rz(-0.0655078478393536) q[5];
ry(3.141531489612657) q[6];
rz(0.29834643151895257) q[6];
ry(-1.5698306441628282) q[7];
rz(-0.9659719568561921) q[7];
ry(3.3661247579530595e-05) q[8];
rz(-0.3744464502464988) q[8];
ry(-1.5707964042655955) q[9];
rz(-1.5707970530564808) q[9];
ry(-3.1415906024510893) q[10];
rz(-2.7620696331233856) q[10];
ry(-1.5706694631348759) q[11];
rz(-3.0160927860496827) q[11];
ry(-0.14765831604065158) q[12];
rz(3.1787460551315654e-06) q[12];
ry(-1.5707961067775384) q[13];
rz(-0.001225643362991374) q[13];
ry(1.5707959290590856) q[14];
rz(4.615864224177812e-08) q[14];
ry(1.157847035899123) q[15];
rz(-0.20598939091239735) q[15];
ry(-3.141592297622965) q[16];
rz(2.927191946486636) q[16];
ry(0.44084577087007837) q[17];
rz(-0.35478031408720206) q[17];
ry(-1.4779361487366707) q[18];
rz(-1.5465526482572045e-05) q[18];
ry(1.570796469810558) q[19];
rz(4.2782349960701834e-05) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5686611194814954) q[0];
rz(-0.397645365028128) q[0];
ry(-0.07162847173031572) q[1];
rz(3.136476193021832) q[1];
ry(-1.5708781772319043) q[2];
rz(0.07247805257852258) q[2];
ry(1.5708343617652663) q[3];
rz(1.571523750987474) q[3];
ry(-2.493740223008142) q[4];
rz(0.0012487275783206186) q[4];
ry(-3.141591180252193) q[5];
rz(-2.333350399753274) q[5];
ry(7.488008169076465e-08) q[6];
rz(-1.0137556300073232) q[6];
ry(-3.564380799403466e-07) q[7];
rz(0.5025911151972042) q[7];
ry(2.791541734024638) q[8];
rz(-3.1415924148079606) q[8];
ry(-1.5707997876319546) q[9];
rz(0.5610060436655822) q[9];
ry(1.196238631178418) q[10];
rz(3.0091547107016456) q[10];
ry(-1.5707966715008672) q[11];
rz(-1.570796421675035) q[11];
ry(-1.5707962909572846) q[12];
rz(3.1415916066502763) q[12];
ry(1.5707965198514866) q[13];
rz(8.099680230005901e-06) q[13];
ry(-1.570796208223527) q[14];
rz(1.5708007731039497) q[14];
ry(1.570796581202129) q[15];
rz(3.141592126086592) q[15];
ry(-3.141561412679685) q[16];
rz(1.1276724583130544) q[16];
ry(-3.141592024083229) q[17];
rz(-1.925579523077507) q[17];
ry(-1.6223904584497635) q[18];
rz(3.1415683305430004) q[18];
ry(-0.13457618434257537) q[19];
rz(3.141547948193271) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.00044985134377117925) q[0];
rz(1.9989193905173517) q[0];
ry(-1.5707613996692131) q[1];
rz(-0.0774660747663578) q[1];
ry(-1.5707971874897593) q[2];
rz(1.6012742856940845) q[2];
ry(0.026528108400620785) q[3];
rz(-1.6489907613965205) q[3];
ry(1.5707723456086944) q[4];
rz(-3.111116748852401) q[4];
ry(-8.10524386799455e-05) q[5];
rz(0.28673607731890266) q[5];
ry(-1.5357993625869062) q[6];
rz(0.03047673284469443) q[6];
ry(1.5708203714261417) q[7];
rz(3.0643805138931124) q[7];
ry(-1.570797660989273) q[8];
rz(0.03047573742408097) q[8];
ry(-1.1949594673765773e-06) q[9];
rz(-2.208920261635248) q[9];
ry(1.5706715687635349) q[10];
rz(-1.5403052663694297) q[10];
ry(1.5707963791244024) q[11];
rz(1.493809514404929) q[11];
ry(-1.5707947660735104) q[12];
rz(-1.540324439330841) q[12];
ry(-1.5707939209421777) q[13];
rz(-1.6477760596586177) q[13];
ry(-1.4736224991593758) q[14];
rz(-3.1111370462878942) q[14];
ry(1.5707966497564838) q[15];
rz(-0.0769764856928248) q[15];
ry(-3.5160863024685796e-05) q[16];
rz(2.6881712275163707) q[16];
ry(1.5707811362158437) q[17];
rz(1.494534767455001) q[17];
ry(0.25152813451478817) q[18];
rz(0.03050570344890689) q[18];
ry(-1.5707930028260009) q[19];
rz(-1.6471282219428518) q[19];