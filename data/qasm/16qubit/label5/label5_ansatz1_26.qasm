OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(2.436282371758554) q[0];
rz(1.327791327602028) q[0];
ry(0.16043068833847587) q[1];
rz(-1.7316748463236828) q[1];
ry(-2.9801225866965506) q[2];
rz(-0.13744429087343427) q[2];
ry(0.02631230644545557) q[3];
rz(-1.683272291582967) q[3];
ry(1.1862154897949617) q[4];
rz(-0.972582482550807) q[4];
ry(-1.798328241151161) q[5];
rz(-2.2335982438041526) q[5];
ry(1.6090985832427913) q[6];
rz(0.676027719744857) q[6];
ry(0.011361016269362858) q[7];
rz(1.868089149861082) q[7];
ry(-0.03257288823891264) q[8];
rz(-1.771782481046755) q[8];
ry(0.08872601968611783) q[9];
rz(-0.28275381946494366) q[9];
ry(-1.7371119139819218) q[10];
rz(0.6584042232454665) q[10];
ry(-1.829499370845312) q[11];
rz(-2.2330633221287477) q[11];
ry(0.0011660501547572366) q[12];
rz(-1.2870044451582723) q[12];
ry(2.4125011603440543) q[13];
rz(1.695364544609411) q[13];
ry(1.4021295753281713) q[14];
rz(0.009968218721074784) q[14];
ry(0.9811634160228586) q[15];
rz(2.306694779127039) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1036835613206306) q[0];
rz(-0.6338358501627477) q[0];
ry(2.23196798519177) q[1];
rz(-2.0526706565993624) q[1];
ry(2.077265528783185) q[2];
rz(-0.9949112305483624) q[2];
ry(-0.002010047116209083) q[3];
rz(0.3159437081258866) q[3];
ry(1.3233269974953754) q[4];
rz(0.6753818474893987) q[4];
ry(-0.9458196809906683) q[5];
rz(0.37271955135107593) q[5];
ry(0.17254994985971328) q[6];
rz(2.6700881265143273) q[6];
ry(0.004185014955903732) q[7];
rz(1.6215736799444445) q[7];
ry(2.5483576989548777) q[8];
rz(-1.952786513532017) q[8];
ry(1.7887908498166507) q[9];
rz(2.483587563693818) q[9];
ry(1.147623941909401) q[10];
rz(2.6093578106469817) q[10];
ry(-2.515936511513556) q[11];
rz(-1.5263539380416093) q[11];
ry(3.1390874984228385) q[12];
rz(2.6372769345573395) q[12];
ry(2.28836810588334) q[13];
rz(2.0432852577997846) q[13];
ry(0.29523231441872283) q[14];
rz(1.1983901147666618) q[14];
ry(0.15005718676921564) q[15];
rz(0.3461599863906999) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.12087353961266) q[0];
rz(-0.4852386883305799) q[0];
ry(2.077819352320531) q[1];
rz(2.3235175218739834) q[1];
ry(3.0343126334901473) q[2];
rz(-1.8139608008307393) q[2];
ry(-3.1394851615634556) q[3];
rz(2.181775005854771) q[3];
ry(-2.5034598821173484) q[4];
rz(-1.300254665743636) q[4];
ry(-0.9873464476381972) q[5];
rz(-0.6116815791853867) q[5];
ry(-0.01348524463444605) q[6];
rz(-1.4363949237215827) q[6];
ry(3.1284195101726175) q[7];
rz(-1.4279369231499484) q[7];
ry(-0.003045577361928688) q[8];
rz(0.10477835029598602) q[8];
ry(-3.0350392684947183) q[9];
rz(-2.7189815331575438) q[9];
ry(-1.5443882879661757) q[10];
rz(1.224244871125924) q[10];
ry(0.68075153929391) q[11];
rz(2.524590147715054) q[11];
ry(-3.0736483700683053) q[12];
rz(-2.471589506611587) q[12];
ry(-2.8676059300776533) q[13];
rz(2.430746033553668) q[13];
ry(1.248805739617719) q[14];
rz(0.14842634679171227) q[14];
ry(-2.663657351759529) q[15];
rz(1.323043145938633) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.623178954282129) q[0];
rz(-3.0696167859434254) q[0];
ry(1.4046732599824445) q[1];
rz(-0.05073032158094855) q[1];
ry(2.047863314819602) q[2];
rz(-1.527961317055424) q[2];
ry(-0.0036198057913649187) q[3];
rz(-3.0190935862056) q[3];
ry(-0.998041796483695) q[4];
rz(-1.1916791130090647) q[4];
ry(-2.7105962504542602) q[5];
rz(3.10653458026765) q[5];
ry(0.08110380010417075) q[6];
rz(-2.9861819507939007) q[6];
ry(1.026297491329804) q[7];
rz(1.6898581909910604) q[7];
ry(-0.9701787709812919) q[8];
rz(0.17802952484042223) q[8];
ry(-0.43196011926836775) q[9];
rz(-2.3860336897894747) q[9];
ry(-0.8549758806556709) q[10];
rz(-0.06199616479347908) q[10];
ry(-2.8383256641839107) q[11];
rz(-1.3480440645187357) q[11];
ry(2.2503033239258308) q[12];
rz(2.9891661879642517) q[12];
ry(2.156437898233776) q[13];
rz(0.3323402387570722) q[13];
ry(1.9872580746911124) q[14];
rz(2.8508238199148055) q[14];
ry(-2.25955409750119) q[15];
rz(2.867888077434312) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.644468855252862) q[0];
rz(-1.9474574694376308) q[0];
ry(-0.9007278728531315) q[1];
rz(-2.604716418313322) q[1];
ry(1.6679592863301145) q[2];
rz(1.0677371920753362) q[2];
ry(3.140983996812897) q[3];
rz(-1.3065571311677673) q[3];
ry(0.005362966416402771) q[4];
rz(-1.0179911449530814) q[4];
ry(2.659170153213197) q[5];
rz(-0.003918148630739359) q[5];
ry(-1.451549571054164) q[6];
rz(-0.02492088006878676) q[6];
ry(-0.009350813152530613) q[7];
rz(-0.3664096153894567) q[7];
ry(5.0364769895949735e-05) q[8];
rz(2.52030919126564) q[8];
ry(-0.08639088593164779) q[9];
rz(1.4249656739628533) q[9];
ry(-0.5654219419351196) q[10];
rz(1.6968962481248333) q[10];
ry(3.138509656427538) q[11];
rz(0.6856757906568811) q[11];
ry(0.09634230658410825) q[12];
rz(-2.1242941909752004) q[12];
ry(2.9308954766426316) q[13];
rz(3.097628925307809) q[13];
ry(0.06617322767115409) q[14];
rz(-2.998308869829007) q[14];
ry(-2.411036544091114) q[15];
rz(-2.076877442530384) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.733144897252918) q[0];
rz(2.3742158308908716) q[0];
ry(-0.1265994322216013) q[1];
rz(-2.6160953361395904) q[1];
ry(-0.4588779770138393) q[2];
rz(0.7145146134286523) q[2];
ry(0.003954858264871467) q[3];
rz(-1.4423924171057474) q[3];
ry(-1.8066658823822053) q[4];
rz(1.4340156784761786) q[4];
ry(-3.1268220248959016) q[5];
rz(-0.0697715444337527) q[5];
ry(-0.7045546497925611) q[6];
rz(-2.364828892094736) q[6];
ry(1.2575555725920078) q[7];
rz(3.0913723660172554) q[7];
ry(0.9769886805536735) q[8];
rz(0.8261123475259327) q[8];
ry(1.4996114970966883) q[9];
rz(1.3848154255381804) q[9];
ry(-1.4482535731261124) q[10];
rz(3.024734372107942) q[10];
ry(-1.3418328140553262) q[11];
rz(-1.0387483509082998) q[11];
ry(1.817794590824473) q[12];
rz(2.231832113421695) q[12];
ry(-1.6511420656635458) q[13];
rz(0.1267896079637047) q[13];
ry(-0.09556998911496173) q[14];
rz(3.007143402142525) q[14];
ry(-1.58160488619691) q[15];
rz(-0.4509977135541274) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.7216241936342298) q[0];
rz(0.046749282351073916) q[0];
ry(-2.919517531638846) q[1];
rz(-1.2429226168931253) q[1];
ry(2.3073289959770076) q[2];
rz(-2.1908946729241725) q[2];
ry(-0.0013597972788695623) q[3];
rz(3.139980988177767) q[3];
ry(0.6317001757747434) q[4];
rz(-1.2942135419618077) q[4];
ry(0.3972377263679245) q[5];
rz(1.2890556953832395) q[5];
ry(2.9565207583209174) q[6];
rz(1.5188512587463363) q[6];
ry(-0.16789878576526182) q[7];
rz(0.5885518206831667) q[7];
ry(-3.0461639731326167) q[8];
rz(1.140803484345974) q[8];
ry(-0.0021006377755899146) q[9];
rz(-1.6719169987341669) q[9];
ry(3.0791327875980143) q[10];
rz(2.2055118193873127) q[10];
ry(-0.0014252226560049763) q[11];
rz(0.9188936006817575) q[11];
ry(1.6596902304033758) q[12];
rz(1.6031764907033867) q[12];
ry(-0.9912995661375181) q[13];
rz(-1.359907237446508) q[13];
ry(0.5512347760104979) q[14];
rz(-0.48186011637851983) q[14];
ry(-1.7590573359165163) q[15];
rz(-0.8748735577662439) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.8212774773565097) q[0];
rz(-1.3726957811887406) q[0];
ry(-0.17729731033342233) q[1];
rz(-2.29406472258708) q[1];
ry(0.9403920138206824) q[2];
rz(-0.7093597114332607) q[2];
ry(0.07012136959898552) q[3];
rz(2.047928150890142) q[3];
ry(-2.4204579401720285) q[4];
rz(1.3033970341213708) q[4];
ry(0.0010744145548633453) q[5];
rz(1.5434175254370368) q[5];
ry(-0.03611543982861552) q[6];
rz(-0.6948071899988083) q[6];
ry(-0.016184252787668996) q[7];
rz(1.3675241038491137) q[7];
ry(2.338215886101852) q[8];
rz(-2.3408797810865503) q[8];
ry(-1.7754127002967415) q[9];
rz(1.9939871030374823) q[9];
ry(2.9288434621908013) q[10];
rz(-3.0737678912647) q[10];
ry(2.426268696194216) q[11];
rz(-0.039140454389237404) q[11];
ry(-3.1311186608326342) q[12];
rz(1.4191538471438891) q[12];
ry(0.08109222611964612) q[13];
rz(-2.7649271683111722) q[13];
ry(-3.1373483082137774) q[14];
rz(-2.2670601638187002) q[14];
ry(-2.2969735626565906) q[15];
rz(-1.2370924715848661) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.6261662248599429) q[0];
rz(2.461763299222616) q[0];
ry(-0.6559547545420585) q[1];
rz(0.07130505453329052) q[1];
ry(1.768794120560327) q[2];
rz(2.3062934606395813) q[2];
ry(-3.1403615639571822) q[3];
rz(-2.3544109213562123) q[3];
ry(-0.28905386513183196) q[4];
rz(-0.027146731154943815) q[4];
ry(0.2742782584936423) q[5];
rz(-0.7922252646138047) q[5];
ry(-0.8839320384025298) q[6];
rz(-0.2450098396116971) q[6];
ry(3.1261842370225907) q[7];
rz(-2.6096381311692487) q[7];
ry(-3.140389347111133) q[8];
rz(-1.2374712859096262) q[8];
ry(0.008054939315233897) q[9];
rz(1.0960246238836895) q[9];
ry(2.9766489542711168) q[10];
rz(1.4810445803163306) q[10];
ry(-0.0033349881487036003) q[11];
rz(-0.23897883144721885) q[11];
ry(0.14129869801408268) q[12];
rz(0.21611496685853912) q[12];
ry(-0.22655819199122354) q[13];
rz(-0.32338171118591447) q[13];
ry(0.013867129802285137) q[14];
rz(2.0680794302523995) q[14];
ry(-0.5324015216988514) q[15];
rz(2.404771531093864) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.7430845342381759) q[0];
rz(-0.8056793941692958) q[0];
ry(-0.44250630647714123) q[1];
rz(-1.5947532336876937) q[1];
ry(0.07900597284846445) q[2];
rz(-2.045907200002429) q[2];
ry(-0.06262217030365225) q[3];
rz(-0.7411277506683887) q[3];
ry(1.1987990269078228) q[4];
rz(-2.8450494592131417) q[4];
ry(3.138764438315105) q[5];
rz(-0.8580265848979849) q[5];
ry(-0.005390069371581596) q[6];
rz(0.06811554531522734) q[6];
ry(-1.6723978181365098) q[7];
rz(-3.1004348554365877) q[7];
ry(-2.71112199814611) q[8];
rz(0.8960475043413938) q[8];
ry(-0.2215281928523743) q[9];
rz(-3.132050577553519) q[9];
ry(0.008068216862426425) q[10];
rz(1.7229115178734578) q[10];
ry(-2.2643021966441497) q[11];
rz(-1.75558531509939) q[11];
ry(1.2434038827428122) q[12];
rz(2.025024883991704) q[12];
ry(-3.0568900395434166) q[13];
rz(1.745680110791431) q[13];
ry(-0.013965202552342681) q[14];
rz(2.326143152575078) q[14];
ry(2.1991047584637693) q[15];
rz(2.2183825124136387) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.6605979064131997) q[0];
rz(-0.9739135586450082) q[0];
ry(-0.6751056615046391) q[1];
rz(2.0537133192120995) q[1];
ry(1.2345414842654883) q[2];
rz(-1.933774362000642) q[2];
ry(0.12171139566878875) q[3];
rz(-2.4772148542119283) q[3];
ry(0.93060556570418) q[4];
rz(0.010155523211053164) q[4];
ry(-0.7053903306112197) q[5];
rz(2.874000577612404) q[5];
ry(-0.4903234924999094) q[6];
rz(2.7041119020010567) q[6];
ry(-2.9422624507526467) q[7];
rz(0.021325905844386206) q[7];
ry(-1.144340388391677) q[8];
rz(-0.016866461531628598) q[8];
ry(-1.5887021249469324) q[9];
rz(-2.437701884289532) q[9];
ry(1.6905844007843862) q[10];
rz(0.34277304746574827) q[10];
ry(-3.140050864647268) q[11];
rz(1.167858121293916) q[11];
ry(-3.0645122943758194) q[12];
rz(-2.653187624965076) q[12];
ry(-1.6481937438925847) q[13];
rz(-2.809117383189416) q[13];
ry(2.630118827330339) q[14];
rz(0.2105555482898161) q[14];
ry(-1.6772444157238797) q[15];
rz(-3.096225086876027) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.8234779481929602) q[0];
rz(-0.21550601297966454) q[0];
ry(-2.3575081158669464) q[1];
rz(-1.3791327929411983) q[1];
ry(3.132479023766869) q[2];
rz(-2.0148763753218875) q[2];
ry(1.246203301039681) q[3];
rz(-2.4522642420387264) q[3];
ry(-2.464962496353218) q[4];
rz(-0.0029412239677301553) q[4];
ry(-0.44749398137096336) q[5];
rz(0.9302910089294544) q[5];
ry(-3.118499603199798) q[6];
rz(2.6637499348204554) q[6];
ry(2.3397012221189146) q[7];
rz(-3.1299878336210227) q[7];
ry(0.20401866008563818) q[8];
rz(-0.1352227132541088) q[8];
ry(-0.00489461541702596) q[9];
rz(-0.07076079810274384) q[9];
ry(3.076624193128873) q[10];
rz(2.60934487964252) q[10];
ry(2.0750496877114655) q[11];
rz(-2.0500282716584612) q[11];
ry(-2.084767644280709) q[12];
rz(0.39344560095803605) q[12];
ry(3.1383321557649375) q[13];
rz(-2.843874905028157) q[13];
ry(1.3874483147916488) q[14];
rz(2.6502998388819674) q[14];
ry(-2.720206387650116) q[15];
rz(-1.3640631083163886) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.449749822873275) q[0];
rz(1.0966458268493229) q[0];
ry(1.9100615739832463) q[1];
rz(-1.5444885395008403) q[1];
ry(-1.5836071397379285) q[2];
rz(-3.1127793946612767) q[2];
ry(3.087549072855745) q[3];
rz(0.8665059975050747) q[3];
ry(-1.5881023944767116) q[4];
rz(-0.05049217268551422) q[4];
ry(3.135164027543147) q[5];
rz(2.193827760563718) q[5];
ry(0.789765969211783) q[6];
rz(-0.9237967019814644) q[6];
ry(1.401191539512456) q[7];
rz(1.3184553721411947) q[7];
ry(3.1289446267501266) q[8];
rz(-0.3433418994587294) q[8];
ry(-0.058169699309226885) q[9];
rz(-2.1054880702159124) q[9];
ry(3.1013464650279414) q[10];
rz(-1.4615642264916178) q[10];
ry(-3.1400575708468783) q[11];
rz(-2.498847582335266) q[11];
ry(-3.140315423023832) q[12];
rz(1.9427833612929657) q[12];
ry(3.094509998114646) q[13];
rz(-1.5620395671607232) q[13];
ry(2.5752993156422272) q[14];
rz(1.5852342524755487) q[14];
ry(-2.4990987112925915) q[15];
rz(-0.3129077041530115) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5792162744045057) q[0];
rz(-1.5197650731950092) q[0];
ry(-0.062161994673336045) q[1];
rz(-0.09453594925238563) q[1];
ry(2.6803845436927585) q[2];
rz(0.008293352459725078) q[2];
ry(2.6298504270789405) q[3];
rz(2.8131864219474787) q[3];
ry(2.3727929851654195) q[4];
rz(1.596775589995251) q[4];
ry(0.06027968232116798) q[5];
rz(-0.4583695786016513) q[5];
ry(0.04895877771449264) q[6];
rz(-2.4318802256823777) q[6];
ry(-1.6391768982347124) q[7];
rz(0.10875211444011423) q[7];
ry(2.6705940145080804) q[8];
rz(0.39477005960732914) q[8];
ry(0.5608233028718894) q[9];
rz(3.0950656646348924) q[9];
ry(1.4134925409040706) q[10];
rz(1.506126755515825) q[10];
ry(-2.553388655139142) q[11];
rz(2.5117579627988125) q[11];
ry(-0.7525085359716526) q[12];
rz(-2.281994219036223) q[12];
ry(-3.135253677556383) q[13];
rz(1.3402314534264415) q[13];
ry(3.132381230355937) q[14];
rz(2.2938458430989703) q[14];
ry(-2.9788650324688275) q[15];
rz(-0.20924326031556814) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.1670367486574458) q[0];
rz(2.9260482866123425) q[0];
ry(0.3852743223657198) q[1];
rz(0.37356084879140883) q[1];
ry(1.0819299098656812) q[2];
rz(-3.1260351863329925) q[2];
ry(-3.0661311020481126) q[3];
rz(-1.3041047915591035) q[3];
ry(-0.005146159541387086) q[4];
rz(1.884358923543733) q[4];
ry(-0.06520613365062466) q[5];
rz(-2.79337689261632) q[5];
ry(-0.006455788542516254) q[6];
rz(-2.926722979985586) q[6];
ry(-3.1328670074074876) q[7];
rz(-0.8758140738193496) q[7];
ry(-0.014333608165960491) q[8];
rz(-0.601970060543094) q[8];
ry(3.1371574560898234) q[9];
rz(-1.6222166594950505) q[9];
ry(-3.053615768167938) q[10];
rz(0.3488014739446452) q[10];
ry(7.15069043282162e-05) q[11];
rz(-0.09478879067879475) q[11];
ry(-1.4862234851257812) q[12];
rz(0.8029153979109322) q[12];
ry(0.04730773742759022) q[13];
rz(1.3037991824312267) q[13];
ry(-0.4612995824224299) q[14];
rz(-1.5553501906965677) q[14];
ry(0.7110993735155775) q[15];
rz(3.0164000288955153) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.7436216624018499) q[0];
rz(-0.6956596940175618) q[0];
ry(-0.05473553827362824) q[1];
rz(-2.203454528911327) q[1];
ry(2.365424658709447) q[2];
rz(-3.1219931432957067) q[2];
ry(-2.9141787430042574) q[3];
rz(2.3121185975858904) q[3];
ry(-3.0438395389146176) q[4];
rz(-2.765420381030667) q[4];
ry(-3.0787544122401465) q[5];
rz(1.1509003921618741) q[5];
ry(-1.5305415173788761) q[6];
rz(3.1047479667671123) q[6];
ry(-1.7455676266091835) q[7];
rz(-1.3549783643426618) q[7];
ry(-1.1278846923214338) q[8];
rz(-3.0474437294741077) q[8];
ry(1.474370746262307) q[9];
rz(-2.500664463749306) q[9];
ry(-0.6304211053398756) q[10];
rz(-1.9685525527663417) q[10];
ry(-1.5298653492168242) q[11];
rz(-1.5702106102986289) q[11];
ry(3.126188508910125) q[12];
rz(-2.7787751543025627) q[12];
ry(3.1328669662938933) q[13];
rz(0.9921728271207773) q[13];
ry(3.0884434851043734) q[14];
rz(2.8658164717301866) q[14];
ry(-1.0077848610142128) q[15];
rz(2.8991532481526723) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.99712131501273) q[0];
rz(-0.6324340210239239) q[0];
ry(-1.00510669925883) q[1];
rz(-2.9982815048174394) q[1];
ry(2.633305282538126) q[2];
rz(3.1406868535989507) q[2];
ry(-2.071235765404797) q[3];
rz(1.5459909953327715) q[3];
ry(-1.4003181051445663) q[4];
rz(-3.138486516817558) q[4];
ry(1.5264176188534098) q[5];
rz(0.0017775419217305939) q[5];
ry(-1.266034993846715) q[6];
rz(-0.0010564408859918828) q[6];
ry(1.5688373082935465) q[7];
rz(3.1401556074028707) q[7];
ry(1.586971059727353) q[8];
rz(3.141326064908288) q[8];
ry(-3.131936126111683) q[9];
rz(-3.077362018704673) q[9];
ry(-1.695561219257517) q[10];
rz(3.1211143259847036) q[10];
ry(-2.6048574374852227) q[11];
rz(-1.4660351017292799) q[11];
ry(-2.3122860783024883) q[12];
rz(-1.952864082354278) q[12];
ry(0.9423773424508425) q[13];
rz(-3.124457705125718) q[13];
ry(-0.9960480632129247) q[14];
rz(1.656191007233117) q[14];
ry(1.7549525824027947) q[15];
rz(-2.4015205728169464) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.0291144303775517) q[0];
rz(-0.05490618326765607) q[0];
ry(1.6002398930900803) q[1];
rz(-0.010956977054700447) q[1];
ry(1.7606592423253957) q[2];
rz(-0.0008269508221561138) q[2];
ry(-3.1364935753925205) q[3];
rz(1.559806452324345) q[3];
ry(2.183991056446203) q[4];
rz(-0.00012516012084340392) q[4];
ry(1.5460628717253468) q[5];
rz(-0.5233573966985414) q[5];
ry(-1.5985471136033407) q[6];
rz(-8.893570038320442e-05) q[6];
ry(-1.5563774270069048) q[7];
rz(1.490930969775568) q[7];
ry(1.8487463793979262) q[8];
rz(-0.028445312310984683) q[8];
ry(0.03060631807996561) q[9];
rz(-3.1274242874896934) q[9];
ry(0.05091017286047261) q[10];
rz(-0.27602801834604207) q[10];
ry(0.001106037404165667) q[11];
rz(1.414179299406979) q[11];
ry(1.5303145876998396) q[12];
rz(-1.4958354387313566) q[12];
ry(0.0020699518372664367) q[13];
rz(1.5623149039091777) q[13];
ry(-3.140101473318236) q[14];
rz(2.6109713370540364) q[14];
ry(0.1962916210490674) q[15];
rz(-1.1918144623571463) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.2499573742977628) q[0];
rz(0.026421056069397316) q[0];
ry(-2.467962805044578) q[1];
rz(-2.6993574615176494) q[1];
ry(0.2880583640107714) q[2];
rz(3.1403435761630876) q[2];
ry(-1.2036113829457724) q[3];
rz(-0.0006302723428879276) q[3];
ry(-1.4217531093405977) q[4];
rz(1.2083167845577008) q[4];
ry(-3.1384588706408127) q[5];
rz(2.6195813549322557) q[5];
ry(1.573224061161148) q[6];
rz(-2.997105765936816) q[6];
ry(3.140352646474016) q[7];
rz(-1.6603161799738384) q[7];
ry(-3.027152430550094) q[8];
rz(3.1122272750247038) q[8];
ry(-1.1118079539209846) q[9];
rz(-1.9341432399057545) q[9];
ry(-1.1886011186627519) q[10];
rz(0.27227227564462914) q[10];
ry(3.14147512977882) q[11];
rz(1.4067658301238426) q[11];
ry(1.4392198007656454) q[12];
rz(2.4067837967220798) q[12];
ry(-3.099739474586087) q[13];
rz(-0.13050028052596951) q[13];
ry(-2.781350291445294) q[14];
rz(1.4321330896725566) q[14];
ry(-1.9050034308487294) q[15];
rz(0.730527549766661) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.9043061400482135) q[0];
rz(0.025691108556018527) q[0];
ry(-0.01185585981167847) q[1];
rz(-0.3658671687617066) q[1];
ry(-0.5834932777539752) q[2];
rz(-3.1404932400124745) q[2];
ry(-1.6016711212450139) q[3];
rz(3.017826524919365) q[3];
ry(0.00019438959916744634) q[4];
rz(1.93365181257606) q[4];
ry(-2.0787996287392376) q[5];
rz(-3.1366696937277383) q[5];
ry(0.0002977035788064254) q[6];
rz(3.0075288900607404) q[6];
ry(1.296257999041868) q[7];
rz(-3.077959981182053) q[7];
ry(-0.581639235832564) q[8];
rz(0.9823922611571863) q[8];
ry(-3.138986978602612) q[9];
rz(-0.39360568072818225) q[9];
ry(-1.5468961479758674) q[10];
rz(3.1115113303669513) q[10];
ry(3.140432546781072) q[11];
rz(-0.33966519049187727) q[11];
ry(2.0737761417178824) q[12];
rz(0.0670617176078358) q[12];
ry(-3.138646615986271) q[13];
rz(-1.3220275718837549) q[13];
ry(-0.0017967460060802068) q[14];
rz(0.010636382781642136) q[14];
ry(3.043636563126327) q[15];
rz(2.0715905916110016) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.5737268203010686) q[0];
rz(-3.1311968478406045) q[0];
ry(-0.019010156468317163) q[1];
rz(3.0229639709894274) q[1];
ry(1.7232349370188986) q[2];
rz(-3.1412490185104653) q[2];
ry(0.001681611996866117) q[3];
rz(-3.017638726812888) q[3];
ry(-1.116214450432724) q[4];
rz(-3.1413468332634333) q[4];
ry(1.490077969799765) q[5];
rz(-0.3523703581698481) q[5];
ry(-2.8314028936724487) q[6];
rz(2.5874907232772304) q[6];
ry(-0.698440262062892) q[7];
rz(-2.92765970646163) q[7];
ry(0.012753285426592351) q[8];
rz(1.1384514037942142) q[8];
ry(0.004197603969698545) q[9];
rz(-1.5255059922216003) q[9];
ry(1.0451086470560886) q[10];
rz(-1.4684149058881177) q[10];
ry(-3.141272437127547) q[11];
rz(0.9549073946542982) q[11];
ry(1.5718045926271582) q[12];
rz(-3.105615987756621) q[12];
ry(3.1415880085802286) q[13];
rz(-0.36815905197024884) q[13];
ry(-2.2841928560675213) q[14];
rz(-1.5532244387555858) q[14];
ry(-2.6442099130589063) q[15];
rz(-2.078895212550183) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.3640601373731647) q[0];
rz(-0.9893918426426761) q[0];
ry(-1.4687087849511142) q[1];
rz(0.5196069232961403) q[1];
ry(2.4273219581561856) q[2];
rz(3.1379840696458947) q[2];
ry(-2.393278542457592) q[3];
rz(3.1078521803157715) q[3];
ry(2.5155998504293025) q[4];
rz(-0.0033089739095428783) q[4];
ry(-3.082181605059673) q[5];
rz(1.9156366171603956) q[5];
ry(3.1407861911526713) q[6];
rz(2.5732666407088667) q[6];
ry(3.1035276682906567) q[7];
rz(0.293070319884203) q[7];
ry(3.141515961148498) q[8];
rz(-1.169669467093859) q[8];
ry(-3.032554316281022) q[9];
rz(2.9678394843155584) q[9];
ry(-0.32438646216494327) q[10];
rz(2.9167907306259493) q[10];
ry(-3.1376934276101776) q[11];
rz(-2.038022004376291) q[11];
ry(-1.091494132509891) q[12];
rz(-0.05501453460671524) q[12];
ry(0.009207086325093705) q[13];
rz(-1.3956241811145826) q[13];
ry(1.531763830738556) q[14];
rz(1.631235205603426) q[14];
ry(2.3734082529871823) q[15];
rz(-1.7389538671953388) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.039189714343215123) q[0];
rz(-1.9749406932471905) q[0];
ry(-1.8044422435224325) q[1];
rz(-2.9891989635532163) q[1];
ry(-1.9391054601777953) q[2];
rz(0.048715330700732806) q[2];
ry(-0.6685312369521871) q[3];
rz(1.8030283274778736) q[3];
ry(-0.49473289376525376) q[4];
rz(-3.133634753110893) q[4];
ry(0.004585505319989913) q[5];
rz(2.7719775505041793) q[5];
ry(0.6225883469981941) q[6];
rz(0.005442543726905034) q[6];
ry(-2.2786621551497355) q[7];
rz(3.1186858404865174) q[7];
ry(0.43391880494380164) q[8];
rz(-2.6884472226786285) q[8];
ry(-1.1199318161092506) q[9];
rz(2.859627481476623) q[9];
ry(-1.3121740038110756) q[10];
rz(2.4224760170802258) q[10];
ry(-3.704554577971692e-05) q[11];
rz(-1.4909496962863205) q[11];
ry(1.4215789909265466) q[12];
rz(1.2136777358592201) q[12];
ry(-1.4844967432568934) q[13];
rz(3.0921582771560194) q[13];
ry(1.5428304365378702) q[14];
rz(-2.204713633616146) q[14];
ry(2.981141030347741) q[15];
rz(1.1303308068690798) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(3.137768368433073) q[0];
rz(0.17374179688041327) q[0];
ry(2.8988116352097655) q[1];
rz(2.911844618784174) q[1];
ry(3.134438223916541) q[2];
rz(-3.0852250240008754) q[2];
ry(-0.002999951562818908) q[3];
rz(1.4026048565586358) q[3];
ry(3.11629003436278) q[4];
rz(0.0008823672622835086) q[4];
ry(0.008175583797706682) q[5];
rz(1.2393190253377293) q[5];
ry(0.7998256232464591) q[6];
rz(-3.139982221596901) q[6];
ry(-0.09298886139333894) q[7];
rz(0.05486321409779946) q[7];
ry(-0.0025176641227828487) q[8];
rz(2.71243083327189) q[8];
ry(3.1389180273670756) q[9];
rz(2.441180653251323) q[9];
ry(3.1389833935996325) q[10];
rz(1.8935865186251624) q[10];
ry(1.5702585760837504) q[11];
rz(0.7454342336453774) q[11];
ry(3.1347374537146515) q[12];
rz(-1.3066284591245239) q[12];
ry(0.07556668933696438) q[13];
rz(1.488099435782516) q[13];
ry(-0.006798754560395358) q[14];
rz(-2.0983637939418776) q[14];
ry(1.3683789307737282) q[15];
rz(2.2113801659803087) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.5998337289885471) q[0];
rz(-1.5603872136195562) q[0];
ry(1.1418994233688267) q[1];
rz(2.050574756014372) q[1];
ry(-2.713300668921211) q[2];
rz(-3.1342659299343087) q[2];
ry(0.5236362165446584) q[3];
rz(-0.04637240769095197) q[3];
ry(-2.685866656361811) q[4];
rz(-0.00436435141577185) q[4];
ry(1.5364633673014811) q[5];
rz(6.144707395350224e-06) q[5];
ry(-2.8403536082992) q[6];
rz(0.0018230246437962452) q[6];
ry(3.1292484020260356) q[7];
rz(0.01031426668513848) q[7];
ry(-2.835159570702291) q[8];
rz(-0.03275941878877126) q[8];
ry(0.48408957712903344) q[9];
rz(-0.8900188802346634) q[9];
ry(-1.5712549022524485) q[10];
rz(-0.38081342522721856) q[10];
ry(-3.140405826709839) q[11];
rz(0.05641618845227647) q[11];
ry(-3.141541988679792) q[12];
rz(-2.62639841270512) q[12];
ry(-0.5777813769646488) q[13];
rz(1.1750771552623147) q[13];
ry(1.4778449145438408) q[14];
rz(-3.0036789795611614) q[14];
ry(1.205511594487176) q[15];
rz(-2.4276896016907323) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.138611196771495) q[0];
rz(-1.4761525421290942) q[0];
ry(3.1325973933083153) q[1];
rz(-1.6394666828136302) q[1];
ry(1.3383177367006758) q[2];
rz(0.04996297878739416) q[2];
ry(-2.532250335025818) q[3];
rz(3.139350414140067) q[3];
ry(1.9270358975499353) q[4];
rz(0.00047735394262021265) q[4];
ry(2.069613252712112) q[5];
rz(0.001144328425698729) q[5];
ry(-1.5777723461774844) q[6];
rz(3.141183713887703) q[6];
ry(-2.735027758605729) q[7];
rz(3.1209542092758933) q[7];
ry(-0.104817100264387) q[8];
rz(-1.4536129740029387) q[8];
ry(-1.4627300799823244) q[9];
rz(0.6331562856741972) q[9];
ry(-3.1409369757345966) q[10];
rz(-2.968703450152386) q[10];
ry(-0.5640461959353571) q[11];
rz(0.8995114656209307) q[11];
ry(-1.5721653746472306) q[12];
rz(0.40951096174996415) q[12];
ry(-1.5765126347138891) q[13];
rz(-3.048774311248915) q[13];
ry(2.9640536692810837) q[14];
rz(-1.442579570861728) q[14];
ry(0.02858103265925518) q[15];
rz(-0.8438048560997106) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.2066796128574877) q[0];
rz(-1.7021347499408188) q[0];
ry(1.7883320826506823) q[1];
rz(-0.1638164884385615) q[1];
ry(-3.0866111810731605) q[2];
rz(-0.3847699745036722) q[2];
ry(-0.8575282058786957) q[3];
rz(-1.6712619650128706) q[3];
ry(-1.2022302829349665) q[4];
rz(3.042772251786762) q[4];
ry(-1.916262140736535) q[5];
rz(0.7463183485676628) q[5];
ry(1.2429902309507055) q[6];
rz(2.874006310729386) q[6];
ry(0.026068629267533616) q[7];
rz(-3.0391498510054196) q[7];
ry(3.1411936178296527) q[8];
rz(1.7312735231621827) q[8];
ry(0.00241526044375303) q[9];
rz(-0.6335679239631807) q[9];
ry(-3.1411942998705937) q[10];
rz(-2.5887348138463917) q[10];
ry(-1.570508241937227) q[11];
rz(1.571498082978434) q[11];
ry(-3.1412403666397526) q[12];
rz(-2.34134760582828) q[12];
ry(-3.134298571651341) q[13];
rz(1.6652604139524136) q[13];
ry(1.572626365379697) q[14];
rz(-2.9450402412840773) q[14];
ry(-1.54709828588583) q[15];
rz(2.4010312982424007) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.8594935910326075) q[0];
rz(-0.04693020137656511) q[0];
ry(0.08370693785059068) q[1];
rz(1.487122928687498) q[1];
ry(1.021124673840922) q[2];
rz(-0.26285487288357695) q[2];
ry(1.3565234189415767) q[3];
rz(1.2843281615798183) q[3];
ry(-0.6807761631794904) q[4];
rz(-0.913706984297935) q[4];
ry(2.7992827284870327) q[5];
rz(-1.1084242013359233) q[5];
ry(0.41581582768224873) q[6];
rz(0.3039944242330223) q[6];
ry(-3.1222244181012924) q[7];
rz(-2.9686440946135586) q[7];
ry(-1.2014180965087802) q[8];
rz(-3.1403637339661596) q[8];
ry(1.4629243427550316) q[9];
rz(-0.9673159185451193) q[9];
ry(1.5706264655224151) q[10];
rz(-1.5704273127042891) q[10];
ry(1.5705475198771404) q[11];
rz(2.2418117541257416) q[11];
ry(0.0007102382429155212) q[12];
rz(-1.4066928315832055) q[12];
ry(1.5692897223659203) q[13];
rz(1.4338164618698381) q[13];
ry(1.5791558023895167) q[14];
rz(-1.5687872888915608) q[14];
ry(-0.0004451135557369911) q[15];
rz(0.8726012687491655) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.07804123847893006) q[0];
rz(0.3643921827645531) q[0];
ry(-3.134492741244731) q[1];
rz(-0.24971251089286156) q[1];
ry(-3.141099609800936) q[2];
rz(1.069334932184977) q[2];
ry(-0.000669219275527779) q[3];
rz(-0.8418422094738922) q[3];
ry(-0.00013341498386143513) q[4];
rz(2.5610674564880327) q[4];
ry(-3.1411241522744233) q[5];
rz(2.887960566538078) q[5];
ry(-3.1397253179002935) q[6];
rz(1.6288468346295124) q[6];
ry(-0.013656072186317566) q[7];
rz(-1.699235501681701) q[7];
ry(0.31230752528747896) q[8];
rz(1.5968530765457194) q[8];
ry(-0.0001932675566193643) q[9];
rz(2.227775819812531) q[9];
ry(1.572646193400652) q[10];
rz(1.8685238250143303) q[10];
ry(-1.5709116683417346) q[11];
rz(0.00012720780316134389) q[11];
ry(6.614221693901357e-05) q[12];
rz(2.5919918290488164) q[12];
ry(-3.1407936064955666) q[13];
rz(-1.1840235279841274) q[13];
ry(-1.568188776020675) q[14];
rz(0.21616658468385988) q[14];
ry(-3.11387052242454) q[15];
rz(-2.4729817019422935) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-2.4051564323265358) q[0];
rz(-0.6123145567885642) q[0];
ry(-1.5560485013143461) q[1];
rz(0.12914258917172725) q[1];
ry(1.938058688263018) q[2];
rz(0.0009314928050949689) q[2];
ry(2.906853586349695) q[3];
rz(-0.06674629587043944) q[3];
ry(-1.508836764191991) q[4];
rz(-2.893474628224497) q[4];
ry(1.800666893538448) q[5];
rz(0.010827946633091945) q[5];
ry(1.6781249152428845) q[6];
rz(2.8739826106177984) q[6];
ry(1.5692147938879353) q[7];
rz(-1.5914456553046528) q[7];
ry(1.61554888694601) q[8];
rz(1.190366013865016) q[8];
ry(1.570039975209455) q[9];
rz(-3.10824361318625) q[9];
ry(0.0020801065155800135) q[10];
rz(1.994773031087962) q[10];
ry(1.570280988708764) q[11];
rz(-3.077845847834601) q[11];
ry(-1.570820876820164) q[12];
rz(-2.4267007903311395) q[12];
ry(-0.00623942097324548) q[13];
rz(2.6584341803360654) q[13];
ry(0.14268518184631498) q[14];
rz(3.0281333116274856) q[14];
ry(2.8916681561551756) q[15];
rz(-1.6147398992506465) q[15];