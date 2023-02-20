OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.172345478631292) q[0];
rz(2.4871200195242538) q[0];
ry(-1.4392021815753868) q[1];
rz(-0.21158015899955895) q[1];
ry(-3.0669161926602353) q[2];
rz(0.2897681587600464) q[2];
ry(0.15770498886757434) q[3];
rz(0.6962696266603086) q[3];
ry(0.5404331998680894) q[4];
rz(-2.0975090256550137) q[4];
ry(1.429109949612478) q[5];
rz(0.6856667580545016) q[5];
ry(3.1395610558387497) q[6];
rz(2.227092958638919) q[6];
ry(-0.001960297075638273) q[7];
rz(1.456622398160273) q[7];
ry(3.1312731540113914) q[8];
rz(0.08534139728092724) q[8];
ry(-1.599867757777223) q[9];
rz(-1.325540462336702) q[9];
ry(0.09706504933560162) q[10];
rz(2.6999589524526364) q[10];
ry(-0.16625248083900424) q[11];
rz(-1.5219291351576931) q[11];
ry(1.4303702141179144) q[12];
rz(-1.2677764658160209) q[12];
ry(-1.467522933948277) q[13];
rz(0.2866139688328749) q[13];
ry(0.9026766498500587) q[14];
rz(2.5391853329632057) q[14];
ry(-0.56962781578954) q[15];
rz(-2.4286984499026776) q[15];
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
ry(-2.8945397321629103) q[0];
rz(2.912570661003236) q[0];
ry(-1.177071238326083) q[1];
rz(-0.474573100458703) q[1];
ry(0.062044045723101765) q[2];
rz(-2.621298899326345) q[2];
ry(2.9663261525510234) q[3];
rz(-0.2772110619947217) q[3];
ry(1.2420167985546415) q[4];
rz(-1.2986796830579979) q[4];
ry(0.056987676263471246) q[5];
rz(-2.7363183394900226) q[5];
ry(-0.0015409933344336935) q[6];
rz(-1.4614452917546936) q[6];
ry(-3.140730370437445) q[7];
rz(-2.3526833251635475) q[7];
ry(-1.5296153613880452) q[8];
rz(0.9599104449052607) q[8];
ry(0.09061924401342884) q[9];
rz(1.3361146276206501) q[9];
ry(3.1315563538964204) q[10];
rz(-0.9983983333728536) q[10];
ry(3.14015982950034) q[11];
rz(-0.651259509348205) q[11];
ry(-1.098334590252635) q[12];
rz(1.8172525705814468) q[12];
ry(1.433575403921641) q[13];
rz(-0.3189227760630057) q[13];
ry(1.2533529742275311) q[14];
rz(0.7462400038628783) q[14];
ry(2.5776661843882422) q[15];
rz(2.1258196814374184) q[15];
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
ry(2.9405256961196353) q[0];
rz(0.6591538565082847) q[0];
ry(-2.722637833181865) q[1];
rz(0.7254936590476629) q[1];
ry(-1.4052754510547656) q[2];
rz(-2.6774135429771926) q[2];
ry(-1.4458397371792702) q[3];
rz(-3.0950073084291736) q[3];
ry(2.8005628943892886) q[4];
rz(1.5546445261268147) q[4];
ry(-1.474329582614013) q[5];
rz(-3.053909190353686) q[5];
ry(-0.03330923716039891) q[6];
rz(-2.9712840375474077) q[6];
ry(-2.9116874588739976) q[7];
rz(-0.8813718294694942) q[7];
ry(3.137462084326997) q[8];
rz(-0.9990415869694695) q[8];
ry(-1.5629597691144506) q[9];
rz(0.6905924683320954) q[9];
ry(0.9452340117012604) q[10];
rz(0.8291799691794018) q[10];
ry(-0.27208193429092936) q[11];
rz(-1.440763460725484) q[11];
ry(1.313429790461565) q[12];
rz(1.0598849757806643) q[12];
ry(-1.44149131533237) q[13];
rz(-1.763974542055447) q[13];
ry(1.297584822004982) q[14];
rz(-1.1753080058026715) q[14];
ry(-0.6048268241170609) q[15];
rz(-0.5116878276567074) q[15];
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
ry(0.1579794227606124) q[0];
rz(1.3041420505443542) q[0];
ry(3.013958985099162) q[1];
rz(-0.31518758174859585) q[1];
ry(1.1014732457166754) q[2];
rz(-3.132892370016699) q[2];
ry(2.184036627486911) q[3];
rz(-1.8955191879010158) q[3];
ry(1.4775073533283944) q[4];
rz(1.7372380225522086) q[4];
ry(-3.108380495210347) q[5];
rz(0.19976226971197253) q[5];
ry(0.5043895814554702) q[6];
rz(-0.8097761353795443) q[6];
ry(-0.5194911185002073) q[7];
rz(1.9808446561608646) q[7];
ry(-0.005037493402356268) q[8];
rz(-2.2256048064682536) q[8];
ry(0.004064056627842483) q[9];
rz(2.2419111337969913) q[9];
ry(-2.610153549575169) q[10];
rz(-3.0830024259648456) q[10];
ry(1.1256639158209159) q[11];
rz(-2.5647695776121933) q[11];
ry(-2.226527295408707) q[12];
rz(-1.5684791348965164) q[12];
ry(1.6375935172445926) q[13];
rz(-2.6116067186588316) q[13];
ry(-0.6583020388675891) q[14];
rz(-2.337805630083982) q[14];
ry(-0.5265751745536695) q[15];
rz(1.947810838799411) q[15];
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
ry(0.6525146501945939) q[0];
rz(2.7293336995653577) q[0];
ry(-1.2373709891378113) q[1];
rz(-0.05990151014048273) q[1];
ry(0.2333483220149475) q[2];
rz(0.2228797188268059) q[2];
ry(-1.754570845350014) q[3];
rz(2.156655858991761) q[3];
ry(-0.49062178800737116) q[4];
rz(1.7563145416895352) q[4];
ry(2.4130700740368236) q[5];
rz(1.0162742257979294) q[5];
ry(-1.6706328056272068) q[6];
rz(1.0566524568063602) q[6];
ry(2.2928081963841698) q[7];
rz(0.6705132194650798) q[7];
ry(0.0025702686446561463) q[8];
rz(0.10047196040511781) q[8];
ry(-3.1362325274971887) q[9];
rz(-0.8483869455251121) q[9];
ry(0.5252620998728622) q[10];
rz(-1.3590394521386084) q[10];
ry(2.343132803837807) q[11];
rz(-2.448290113823468) q[11];
ry(-1.227656635520637) q[12];
rz(-0.6838215518864582) q[12];
ry(-0.8249322747175958) q[13];
rz(-2.478296557955594) q[13];
ry(1.0696496206189297) q[14];
rz(1.7520708014020716) q[14];
ry(1.732805333510847) q[15];
rz(2.764175265372182) q[15];
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
ry(-1.1216161955969288) q[0];
rz(-2.054255719625675) q[0];
ry(1.647423381107303) q[1];
rz(-2.433503615945384) q[1];
ry(-2.243550856976537) q[2];
rz(2.5806803897098427) q[2];
ry(0.7563901137501796) q[3];
rz(2.661729853934845) q[3];
ry(2.0805319729283434) q[4];
rz(-1.8212155912473513) q[4];
ry(2.213913038667976) q[5];
rz(-2.9371594807659394) q[5];
ry(-2.2165937073933892) q[6];
rz(-0.7615103556561705) q[6];
ry(0.4241795364592162) q[7];
rz(-0.22810118524474984) q[7];
ry(0.002188399839845978) q[8];
rz(-1.1057925508345334) q[8];
ry(-0.005524383714147585) q[9];
rz(-1.2651411882301797) q[9];
ry(1.7613202543099025) q[10];
rz(-1.798502489213555) q[10];
ry(2.18712971720292) q[11];
rz(-0.2292033420796864) q[11];
ry(0.06965609767829177) q[12];
rz(2.29622365966757) q[12];
ry(1.0952031797212005) q[13];
rz(-1.1016483454146602) q[13];
ry(0.6700209493558962) q[14];
rz(-0.696170690841674) q[14];
ry(-1.261860521799677) q[15];
rz(-0.73214065330742) q[15];
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
ry(-2.9126794782636756) q[0];
rz(-2.436124198512451) q[0];
ry(2.8103963188939005) q[1];
rz(-2.485049180920371) q[1];
ry(2.875602594177293) q[2];
rz(-1.0131946615650174) q[2];
ry(0.06797535745973687) q[3];
rz(3.0570852965402846) q[3];
ry(-1.9640770725103385) q[4];
rz(-0.34478499291088127) q[4];
ry(2.253768139420095) q[5];
rz(-1.7810715957023513) q[5];
ry(-0.45185179782161056) q[6];
rz(1.0177959371682477) q[6];
ry(0.2018062192852392) q[7];
rz(-1.8621454864782878) q[7];
ry(-0.0243408224009789) q[8];
rz(0.70454847792569) q[8];
ry(3.124398917028802) q[9];
rz(1.950960424681739) q[9];
ry(0.8674526631357242) q[10];
rz(0.2154303078604722) q[10];
ry(-2.5344387946269693) q[11];
rz(-2.4535079929068577) q[11];
ry(-0.46455743872544275) q[12];
rz(1.1454273525491434) q[12];
ry(2.093819978539716) q[13];
rz(-2.690841772784621) q[13];
ry(0.7863963561711839) q[14];
rz(2.9924194491224827) q[14];
ry(-0.7185638817719768) q[15];
rz(-2.1327959716941676) q[15];
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
ry(1.4637460802046571) q[0];
rz(3.0339181150974293) q[0];
ry(-2.232425259268598) q[1];
rz(-2.9555402288180104) q[1];
ry(-0.1696019398047313) q[2];
rz(-1.6826381224277611) q[2];
ry(2.9174069049938667) q[3];
rz(1.4186861061774856) q[3];
ry(3.112531749915978) q[4];
rz(-1.0043392352912595) q[4];
ry(-2.204886679983571) q[5];
rz(-0.9964600690228593) q[5];
ry(2.2601198180182367) q[6];
rz(2.538132098157775) q[6];
ry(-2.147536647076603) q[7];
rz(-0.026201235041131407) q[7];
ry(3.1380882410186683) q[8];
rz(2.6170979366301816) q[8];
ry(-3.1406955943025774) q[9];
rz(-0.5110673134131503) q[9];
ry(2.0161748245059607) q[10];
rz(1.037936976517738) q[10];
ry(1.3184152129294284) q[11];
rz(-2.2569310612007527) q[11];
ry(-1.8333447028015506) q[12];
rz(-2.3509155500517713) q[12];
ry(-2.4729693059183777) q[13];
rz(1.0351855377535317) q[13];
ry(0.1317489395399445) q[14];
rz(-2.1192601457031266) q[14];
ry(1.982537825407637) q[15];
rz(-1.887598137888044) q[15];
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
ry(0.8925280834121727) q[0];
rz(2.1183105995983498) q[0];
ry(2.4500873026244565) q[1];
rz(0.6848683371651321) q[1];
ry(2.130055432003102) q[2];
rz(2.9497187262301203) q[2];
ry(1.020647609287728) q[3];
rz(-0.16221210754767412) q[3];
ry(-0.5864736591098786) q[4];
rz(-2.851298832524224) q[4];
ry(-2.5046574703241604) q[5];
rz(0.12002459874610338) q[5];
ry(-1.0747726810145606) q[6];
rz(3.034313745497661) q[6];
ry(1.4154563329572418) q[7];
rz(2.6832755957885643) q[7];
ry(1.5320268921463276) q[8];
rz(2.329017551327399) q[8];
ry(1.5573427152972799) q[9];
rz(-2.0096526635789997) q[9];
ry(-2.8227067300631) q[10];
rz(-0.07538336168155002) q[10];
ry(-2.2917265988631623) q[11];
rz(2.0965100008400492) q[11];
ry(2.93759518503844) q[12];
rz(3.1181111005730435) q[12];
ry(0.551577924875108) q[13];
rz(-1.9510445817035436) q[13];
ry(2.5755001028169446) q[14];
rz(1.9496095012216768) q[14];
ry(-0.08114199673871697) q[15];
rz(-1.2151700183929466) q[15];
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
ry(-2.142349129012474) q[0];
rz(2.8892903870205693) q[0];
ry(-2.835783611680924) q[1];
rz(-1.5057878504762696) q[1];
ry(1.1360361845680542) q[2];
rz(0.4608230801391796) q[2];
ry(-2.0623586423148) q[3];
rz(-0.21966465471632812) q[3];
ry(2.0500780751651893) q[4];
rz(-2.686031163367302) q[4];
ry(0.56547808059148) q[5];
rz(2.772647744524119) q[5];
ry(-0.01113129390867229) q[6];
rz(-0.02120066197103579) q[6];
ry(-3.136141576059933) q[7];
rz(-1.6565816504045416) q[7];
ry(-0.010563244035859398) q[8];
rz(0.7750046695681325) q[8];
ry(0.013301356553180295) q[9];
rz(-1.0706162429560724) q[9];
ry(-0.17563650105850304) q[10];
rz(1.7959632274452586) q[10];
ry(-1.9336995700898234) q[11];
rz(-1.8162542594217879) q[11];
ry(-1.574816340281373) q[12];
rz(-2.308079760609635) q[12];
ry(0.6545149832420486) q[13];
rz(1.0970089554137015) q[13];
ry(-1.3025540695708804) q[14];
rz(-2.664312006859169) q[14];
ry(-0.6427278786786934) q[15];
rz(-1.6389344330489617) q[15];
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
ry(-2.606515778759892) q[0];
rz(-1.274718964353009) q[0];
ry(0.6777097684807074) q[1];
rz(1.952122518675151) q[1];
ry(-0.1724868747598074) q[2];
rz(-0.8445303775035048) q[2];
ry(0.13595246704677066) q[3];
rz(-1.966115802559477) q[3];
ry(2.139797829555606) q[4];
rz(1.7381339483858436) q[4];
ry(-0.8330571366215391) q[5];
rz(-1.2109475631081645) q[5];
ry(1.6834777717009992) q[6];
rz(1.4009047980267162) q[6];
ry(-1.914616292861722) q[7];
rz(1.7427349535948444) q[7];
ry(-1.567889872844723) q[8];
rz(-1.8665253211431316) q[8];
ry(1.5672609757447071) q[9];
rz(-1.562274312295696) q[9];
ry(-0.8700862376176853) q[10];
rz(-2.0839964023483946) q[10];
ry(-1.6379996759683333) q[11];
rz(-0.011085890162919039) q[11];
ry(2.2696983401380457) q[12];
rz(1.846847057987797) q[12];
ry(-0.8801066566571046) q[13];
rz(-0.2741950772957234) q[13];
ry(2.6464927668456273) q[14];
rz(-1.8899381754985227) q[14];
ry(-1.4488307407845245) q[15];
rz(-2.8799262483164654) q[15];
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
ry(0.6498589936678848) q[0];
rz(-2.7019053136459537) q[0];
ry(-1.666903841548236) q[1];
rz(-2.4594157369551213) q[1];
ry(-0.20016151107335303) q[2];
rz(2.662057939232624) q[2];
ry(-2.988380827218044) q[3];
rz(0.11472450022011937) q[3];
ry(0.738338906104242) q[4];
rz(0.3953120039104522) q[4];
ry(0.6335147925489242) q[5];
rz(2.9077929487591496) q[5];
ry(2.6682463661811004) q[6];
rz(2.953573614228662) q[6];
ry(2.7080863750274244) q[7];
rz(-0.5446951551850641) q[7];
ry(0.01105915300082394) q[8];
rz(-3.065853427620988) q[8];
ry(0.017488775116429878) q[9];
rz(-1.5071867562601966) q[9];
ry(0.42694540507510104) q[10];
rz(-0.15455888679880303) q[10];
ry(-3.0418099241494434) q[11];
rz(-0.24575660690265447) q[11];
ry(1.1174687681200695) q[12];
rz(-1.5086639880551589) q[12];
ry(2.2672456715688685) q[13];
rz(-0.966673915642814) q[13];
ry(0.645794198634845) q[14];
rz(-0.0617062495405136) q[14];
ry(2.0393766382267113) q[15];
rz(-2.7666652752273055) q[15];
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
ry(2.474845519513976) q[0];
rz(1.5624168272772434) q[0];
ry(-2.741393959326412) q[1];
rz(1.0280375703632105) q[1];
ry(-1.651749908460734) q[2];
rz(1.2789909479143942) q[2];
ry(1.5067641661970077) q[3];
rz(2.0384173360691293) q[3];
ry(0.7547598339903367) q[4];
rz(-0.6709001027801369) q[4];
ry(-1.3798021875152264) q[5];
rz(1.84147907104603) q[5];
ry(0.6116950882360932) q[6];
rz(-3.0022877103756764) q[6];
ry(-2.4755592354038396) q[7];
rz(-0.7294700135229739) q[7];
ry(3.132769997348848) q[8];
rz(-0.16536946836805339) q[8];
ry(-3.134787947808548) q[9];
rz(-1.7034188838621047) q[9];
ry(0.6225019747127231) q[10];
rz(-2.626319802106527) q[10];
ry(-1.8195193192480765) q[11];
rz(1.9752941864288749) q[11];
ry(1.9249658635447586) q[12];
rz(-1.0566298140357282) q[12];
ry(2.525103410109677) q[13];
rz(1.8794260651132229) q[13];
ry(-1.5247475548290748) q[14];
rz(0.8907909506619529) q[14];
ry(1.6527217433306456) q[15];
rz(1.4360074215091014) q[15];
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
ry(0.6678030927077598) q[0];
rz(-2.4021597427838812) q[0];
ry(2.82621416900186) q[1];
rz(2.151596595729267) q[1];
ry(-0.18444085784611539) q[2];
rz(0.14892323226229912) q[2];
ry(2.9863990309844533) q[3];
rz(-2.9920250597202096) q[3];
ry(-1.4220914482690554) q[4];
rz(-2.3961729554093387) q[4];
ry(-0.670183090126144) q[5];
rz(-1.9221477219405623) q[5];
ry(1.1007885530173547) q[6];
rz(-1.5808649718135783) q[6];
ry(-0.38985993422241005) q[7];
rz(0.06591637305670074) q[7];
ry(3.135628494527813) q[8];
rz(0.313083336164798) q[8];
ry(-3.1340783228967997) q[9];
rz(1.2491113849555084) q[9];
ry(-0.9488013867795617) q[10];
rz(2.0364520212404886) q[10];
ry(1.6552958673388065) q[11];
rz(-2.9589346212257444) q[11];
ry(1.274656747708415) q[12];
rz(1.6322604560279748) q[12];
ry(-1.7543281238034447) q[13];
rz(1.2439854664090708) q[13];
ry(-2.5794965721601377) q[14];
rz(-0.5663402166154841) q[14];
ry(1.3741193382520156) q[15];
rz(1.5151477506502742) q[15];
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
ry(-1.584720408405739) q[0];
rz(-2.1767942480070435) q[0];
ry(-2.1675365906600637) q[1];
rz(-1.4258313324741112) q[1];
ry(-2.763316120051374) q[2];
rz(-3.112842268560936) q[2];
ry(2.7091423916462953) q[3];
rz(-2.9121348700217227) q[3];
ry(-2.206297229410826) q[4];
rz(2.967402586289751) q[4];
ry(-0.3911174536507884) q[5];
rz(0.8489066853399488) q[5];
ry(-2.8014187754140525) q[6];
rz(0.8320654905953893) q[6];
ry(0.32729935950079714) q[7];
rz(0.6041939344510229) q[7];
ry(-1.5722374445636587) q[8];
rz(-1.4696876451504322) q[8];
ry(-3.1395364650566973) q[9];
rz(-0.8210189549561228) q[9];
ry(0.9215904706668905) q[10];
rz(1.6308226569735118) q[10];
ry(0.6367318266504209) q[11];
rz(-2.4386848677360264) q[11];
ry(2.460082055831061) q[12];
rz(2.588347690141841) q[12];
ry(1.447609464792106) q[13];
rz(1.0923630957884167) q[13];
ry(-0.4830377041723124) q[14];
rz(-2.8427824377842974) q[14];
ry(0.569531766943685) q[15];
rz(1.0286528116835978) q[15];
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
ry(2.7031094044059962) q[0];
rz(-2.829852564979277) q[0];
ry(2.0682428751013244) q[1];
rz(-1.5879023108243737) q[1];
ry(-0.07312151941396206) q[2];
rz(-1.790787241028573) q[2];
ry(-3.093796005179066) q[3];
rz(2.7218932571117973) q[3];
ry(2.4199241655576906) q[4];
rz(2.834959434946168) q[4];
ry(-0.43925694090664336) q[5];
rz(-2.586380734415699) q[5];
ry(-1.5689580391370095) q[6];
rz(-2.4730694858624678) q[6];
ry(1.5707502203033004) q[7];
rz(1.574253119413303) q[7];
ry(-3.139544121621247) q[8];
rz(-2.7721950410797382) q[8];
ry(3.1393195654510064) q[9];
rz(1.2581667218864951) q[9];
ry(-1.5708959300345313) q[10];
rz(-1.5691294506764801) q[10];
ry(-1.5707927643504878) q[11];
rz(1.5700514122757034) q[11];
ry(2.08483578972657) q[12];
rz(-0.4867336132041339) q[12];
ry(2.582927669984264) q[13];
rz(2.4107760401177507) q[13];
ry(-0.10229582929555471) q[14];
rz(0.217005001060155) q[14];
ry(1.451994403567776) q[15];
rz(0.594397810142041) q[15];
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
ry(-1.7793824345473714) q[0];
rz(-1.9923583198615133) q[0];
ry(-1.854767738097103) q[1];
rz(-0.11940176307027262) q[1];
ry(1.6598078430994052) q[2];
rz(-3.111459758063398) q[2];
ry(0.03219561461301237) q[3];
rz(-1.7114883065336546) q[3];
ry(2.69528374402923) q[4];
rz(2.325131896469534) q[4];
ry(-2.1107719262224087) q[5];
rz(-2.0954168296928963) q[5];
ry(-0.04056317975950385) q[6];
rz(-2.0069726734902464) q[6];
ry(2.302072785936323) q[7];
rz(-0.2879300763562959) q[7];
ry(3.1397587065744736) q[8];
rz(0.3550676047366365) q[8];
ry(-3.129661205762089) q[9];
rz(1.3147983812137056) q[9];
ry(3.0175338124985354) q[10];
rz(-3.139589222876353) q[10];
ry(-0.2743247136250222) q[11];
rz(0.000959752932581992) q[11];
ry(2.0722878617583858) q[12];
rz(1.3090381138980813) q[12];
ry(0.26094933790975045) q[13];
rz(1.3814999494106948) q[13];
ry(-2.4691114762148585) q[14];
rz(2.974204963312112) q[14];
ry(-0.44243523875935153) q[15];
rz(-1.9765662846447896) q[15];
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
ry(0.4045761155144936) q[0];
rz(-0.3585584533957551) q[0];
ry(1.5420162337450949) q[1];
rz(-2.675375674703679) q[1];
ry(-1.5531802657575584) q[2];
rz(0.1589271058438484) q[2];
ry(-0.0672777272516451) q[3];
rz(-0.241791288353909) q[3];
ry(-0.02454218032713603) q[4];
rz(0.05849052956131118) q[4];
ry(3.1000621043024985) q[5];
rz(-1.2040105174632683) q[5];
ry(0.016202372870787407) q[6];
rz(-1.7893649419981346) q[6];
ry(-2.7078203912963144) q[7];
rz(-1.7376410664381776) q[7];
ry(-0.003062193418680909) q[8];
rz(1.505576653732243) q[8];
ry(3.138139217648951) q[9];
rz(-1.6213769146676524) q[9];
ry(1.5585333968078239) q[10];
rz(1.5702283740340046) q[10];
ry(1.6967758163897138) q[11];
rz(-3.653682508097944e-05) q[11];
ry(-2.489907408072427) q[12];
rz(-2.248451902395818) q[12];
ry(-0.918891561820877) q[13];
rz(0.04553137816651365) q[13];
ry(2.6002908605645216) q[14];
rz(0.6183528534499723) q[14];
ry(1.4395563047771187) q[15];
rz(2.6332099237936717) q[15];
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
ry(1.64862281142406) q[0];
rz(-3.1377998691406956) q[0];
ry(-0.08933233725332991) q[1];
rz(-1.4277769649843393) q[1];
ry(-0.18383637565295038) q[2];
rz(1.3363731432628416) q[2];
ry(-3.119529909284491) q[3];
rz(1.83112646620637) q[3];
ry(3.107697695319527) q[4];
rz(0.3107707828387856) q[4];
ry(-3.1287141332041037) q[5];
rz(-0.8987259804036345) q[5];
ry(1.5931642717999834) q[6];
rz(-1.5899159464523474) q[6];
ry(-1.5713619453016923) q[7];
rz(0.4401592954094351) q[7];
ry(-3.0971473425428018) q[8];
rz(0.025676910322013923) q[8];
ry(-3.0404937967909693) q[9];
rz(-1.5042354880137356) q[9];
ry(-1.563850685932893) q[10];
rz(-2.458530196556953) q[10];
ry(-2.6955290631930073) q[11];
rz(1.5964632785302413) q[11];
ry(0.23834117452951809) q[12];
rz(2.4913946045616213) q[12];
ry(0.5580963718787784) q[13];
rz(0.5878282290296344) q[13];
ry(-0.7117732313135411) q[14];
rz(-2.7148422790401017) q[14];
ry(1.7251003687016953) q[15];
rz(1.7781333067987877) q[15];
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
ry(-1.9864440465908983) q[0];
rz(1.2191475419553044) q[0];
ry(3.112010626925104) q[1];
rz(-0.9936545780749216) q[1];
ry(3.0797756289559732) q[2];
rz(-1.2292889783258758) q[2];
ry(-2.7988062384429653) q[3];
rz(-1.3539145224982059) q[3];
ry(-0.0028930279796645325) q[4];
rz(2.2592952368516874) q[4];
ry(1.6270694486617974) q[5];
rz(1.161697020663416) q[5];
ry(0.8957232899190775) q[6];
rz(2.455497119940539) q[6];
ry(0.011063431591190387) q[7];
rz(0.65636690392977) q[7];
ry(-1.5581115141628012) q[8];
rz(-1.5714450657514512) q[8];
ry(1.569743299439697) q[9];
rz(0.001125811109108117) q[9];
ry(3.1412533303011516) q[10];
rz(-2.606922477862328) q[10];
ry(3.078256481607082) q[11];
rz(-3.1154723163256843) q[11];
ry(0.00043959326627707185) q[12];
rz(-2.8740322935205462) q[12];
ry(1.572336696803676) q[13];
rz(-3.1413294849819584) q[13];
ry(2.0777434173286338) q[14];
rz(0.43692717505242745) q[14];
ry(0.6467851961018605) q[15];
rz(-1.2771150914478753) q[15];
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
ry(-0.18395166675592733) q[0];
rz(-2.6386434081686074) q[0];
ry(1.5522736582949397) q[1];
rz(1.4713641002698221) q[1];
ry(3.1283806517567063) q[2];
rz(-1.1939183916595022) q[2];
ry(0.13019630777141367) q[3];
rz(2.987854795615496) q[3];
ry(-0.011999154163244263) q[4];
rz(-0.6425907240077064) q[4];
ry(3.104315319038849) q[5];
rz(1.1191853970805645) q[5];
ry(-0.00041798987303926145) q[6];
rz(1.9281223450691956) q[6];
ry(-3.1414168514746454) q[7];
rz(0.5120954939773705) q[7];
ry(-1.485063176118816) q[8];
rz(-3.141492707332416) q[8];
ry(1.5637761164744914) q[9];
rz(-0.00028313611122765536) q[9];
ry(3.140376025098587) q[10];
rz(1.5535288479109095) q[10];
ry(-1.5719981619358276) q[11];
rz(1.5608863102302362) q[11];
ry(-0.0005580806754386104) q[12];
rz(-1.3065557425847891) q[12];
ry(1.3989983777682624) q[13];
rz(0.007539822859006929) q[13];
ry(1.5730115089695096) q[14];
rz(1.5706122262680031) q[14];
ry(-1.574397071722558) q[15];
rz(1.571399828332315) q[15];
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
ry(0.5235581966952497) q[0];
rz(2.957789510522366) q[0];
ry(2.746342192604581) q[1];
rz(-3.1390313060992465) q[1];
ry(-1.5624079842778513) q[2];
rz(1.5500688601230053) q[2];
ry(1.5904563263867477) q[3];
rz(0.3100949604781295) q[3];
ry(0.009885065255786252) q[4];
rz(-0.541570602261506) q[4];
ry(-1.6413905701783502) q[5];
rz(-1.560581407305537) q[5];
ry(0.0008095629244142011) q[6];
rz(-0.45638924309813067) q[6];
ry(-3.1414433836785034) q[7];
rz(-2.3306378001714134) q[7];
ry(1.5679099845518252) q[8];
rz(0.20791033193437336) q[8];
ry(1.5718543712733624) q[9];
rz(0.18720959402725246) q[9];
ry(1.5621800176774743) q[10];
rz(-0.0608657907884539) q[10];
ry(-1.601737807370065) q[11];
rz(-0.17630231503847074) q[11];
ry(0.058402064753808554) q[12];
rz(-1.4686014666198755) q[12];
ry(-0.04451077441378714) q[13];
rz(-1.4252328389771405) q[13];
ry(2.076492202810063) q[14];
rz(-1.969972011041491) q[14];
ry(-0.5372620158176975) q[15];
rz(-0.048800510183558465) q[15];
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
ry(1.1754865241739434) q[0];
rz(0.8801507241699235) q[0];
ry(-1.8994167310372585) q[1];
rz(0.6094312271204112) q[1];
ry(-1.2209149412698708) q[2];
rz(-2.251482336778915) q[2];
ry(1.412432832639666) q[3];
rz(-0.9689130052015376) q[3];
ry(-1.003104739497382) q[4];
rz(-2.2494314314292723) q[4];
ry(0.17743380501389885) q[5];
rz(-2.8939654936842274) q[5];
ry(2.387151980114828) q[6];
rz(1.8672282055535616) q[6];
ry(1.8098361134328378) q[7];
rz(1.843610516958772) q[7];
ry(-3.0819940310981515) q[8];
rz(0.5063770020720864) q[8];
ry(3.055464174566765) q[9];
rz(-2.645669586990046) q[9];
ry(1.5621512760537355) q[10];
rz(-1.2710163412775577) q[10];
ry(-1.5849252764141593) q[11];
rz(-1.2622843804763555) q[11];
ry(1.5764236894496984) q[12];
rz(1.8735101400137986) q[12];
ry(3.1301931164395684) q[13];
rz(-1.1090775285717678) q[13];
ry(-0.0020048553040465937) q[14];
rz(-0.8690334236154511) q[14];
ry(2.8443642931849498) q[15];
rz(0.26193437429464433) q[15];