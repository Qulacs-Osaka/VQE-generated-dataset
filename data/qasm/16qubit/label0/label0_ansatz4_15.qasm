OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(2.752570408285996) q[0];
rz(-1.0636744372485483) q[0];
ry(2.3406737806878644) q[1];
rz(-1.4334480458382872) q[1];
ry(0.1553994006641073) q[2];
rz(-2.845556932882547) q[2];
ry(-2.952770394733939) q[3];
rz(-0.9378488833902828) q[3];
ry(-0.025976939782445738) q[4];
rz(2.46890556258112) q[4];
ry(-1.525128452436561) q[5];
rz(3.1394381996284886) q[5];
ry(-3.141588028974174) q[6];
rz(-3.075283277505774) q[6];
ry(-2.363178911757302e-05) q[7];
rz(-2.2749858648100645) q[7];
ry(-1.9259235220154995) q[8];
rz(2.920966292303164) q[8];
ry(0.5375612382436925) q[9];
rz(1.3424697562403372) q[9];
ry(-0.09136469328089361) q[10];
rz(-2.992413049123954) q[10];
ry(2.9657680723505093) q[11];
rz(2.2759400241225873) q[11];
ry(-1.5711966411305616) q[12];
rz(0.9243408861802712) q[12];
ry(-1.4754599198914136) q[13];
rz(3.0304299786695377) q[13];
ry(-1.773610371315196) q[14];
rz(-1.6431740312284169) q[14];
ry(-1.0550116880741902) q[15];
rz(2.628239799549835) q[15];
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
ry(2.2947631170066947) q[0];
rz(2.8801463583705957) q[0];
ry(1.4197549324515275) q[1];
rz(-0.908517702832874) q[1];
ry(-3.1361356333979526) q[2];
rz(-2.574572210756463) q[2];
ry(0.0025803666506200233) q[3];
rz(0.6795906864576906) q[3];
ry(-1.020227895239198) q[4];
rz(1.6302632069584053) q[4];
ry(-1.6060044694928708) q[5];
rz(-1.812237874424354) q[5];
ry(3.1415259038682835) q[6];
rz(2.869484910028321) q[6];
ry(-3.141392309842644) q[7];
rz(0.1367052166239739) q[7];
ry(2.812883708319406) q[8];
rz(0.08556783024457477) q[8];
ry(-1.7588696669434825) q[9];
rz(3.0679040959918837) q[9];
ry(3.1414430246395324) q[10];
rz(-1.7232329427786401) q[10];
ry(2.2840201920892866e-05) q[11];
rz(1.370892920324569) q[11];
ry(3.1396023082685134) q[12];
rz(-0.17772310310229059) q[12];
ry(3.141150919256357) q[13];
rz(0.41845281220043784) q[13];
ry(1.6355113239286574) q[14];
rz(-3.0985000329544197) q[14];
ry(-1.4064999520137167) q[15];
rz(0.2817324846032709) q[15];
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
ry(2.088297669469828) q[0];
rz(2.8458495441902465) q[0];
ry(-0.4309456084742243) q[1];
rz(-2.1792478192453473) q[1];
ry(2.0401160465392785) q[2];
rz(-0.9343507193133948) q[2];
ry(-0.8122159643186393) q[3];
rz(-2.105714080849329) q[3];
ry(-1.4914272825966997) q[4];
rz(1.015222856737318) q[4];
ry(-1.3653743990455045) q[5];
rz(2.974779313817072) q[5];
ry(3.141567009998827) q[6];
rz(-1.984789677936469) q[6];
ry(-3.54782706534943e-05) q[7];
rz(-1.885140228878675) q[7];
ry(2.917596112766022) q[8];
rz(-2.4092062489939) q[8];
ry(-1.6425162797537114) q[9];
rz(-2.0527380935769157) q[9];
ry(-3.034664957729119) q[10];
rz(-2.093288998659336) q[10];
ry(-0.21550621926099622) q[11];
rz(-2.9248451519654277) q[11];
ry(-0.009735527965674784) q[12];
rz(-0.9462237192553623) q[12];
ry(-0.01379039637937348) q[13];
rz(-2.8630135405908153) q[13];
ry(0.9938015341737828) q[14];
rz(1.7485905501103822) q[14];
ry(1.652290376273056) q[15];
rz(3.0349025467837802) q[15];
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
ry(2.73102003440483) q[0];
rz(1.8380387465555685) q[0];
ry(-0.7112694138310485) q[1];
rz(-2.614407924674336) q[1];
ry(-1.564626295240429) q[2];
rz(-3.1367911665630634) q[2];
ry(1.5658526100118868) q[3];
rz(3.1369791638857487) q[3];
ry(1.5662250653446774) q[4];
rz(0.16431697017927976) q[4];
ry(-1.5796672420054463) q[5];
rz(-0.9850166709176148) q[5];
ry(0.004829246886402849) q[6];
rz(-2.246632537005432) q[6];
ry(3.1332306929541827) q[7];
rz(-2.0550697126481516) q[7];
ry(0.04971863278204558) q[8];
rz(-2.2481420982726172) q[8];
ry(2.5724383945487186) q[9];
rz(2.3538329140531227) q[9];
ry(3.1346081840971554) q[10];
rz(-0.3684288051441822) q[10];
ry(-3.139963488835302) q[11];
rz(-3.0084501461497415) q[11];
ry(3.1377803337891677) q[12];
rz(-0.49382137825364836) q[12];
ry(-0.0017344483559087974) q[13];
rz(-1.9690628717452316) q[13];
ry(-1.0571846958607107) q[14];
rz(-1.0822880498861618) q[14];
ry(-0.75010218509024) q[15];
rz(-1.7087354256139882) q[15];
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
ry(-1.9418798572708262) q[0];
rz(0.040466465753487904) q[0];
ry(-2.0877516631665873) q[1];
rz(2.2565774900756175) q[1];
ry(1.5770048581950658) q[2];
rz(2.6895865580012392) q[2];
ry(-1.5750452962190193) q[3];
rz(2.250791301667016) q[3];
ry(-0.07960732999621456) q[4];
rz(-0.7592317760828772) q[4];
ry(-0.001450326063387449) q[5];
rz(-0.5605480259472265) q[5];
ry(-3.141552068730482) q[6];
rz(-1.518938028018848) q[6];
ry(-3.141475877678819) q[7];
rz(0.4864292620978288) q[7];
ry(-0.880913818926413) q[8];
rz(-1.8951375089938072) q[8];
ry(-0.8927813138176353) q[9];
rz(-1.8335219283014448) q[9];
ry(-1.6569089429110884) q[10];
rz(-1.8778019153812386) q[10];
ry(-1.7327237481868192) q[11];
rz(2.6715251400796247) q[11];
ry(-1.5374288616269158) q[12];
rz(-1.7424188246313925) q[12];
ry(-3.113830283706674) q[13];
rz(1.9742487322196232) q[13];
ry(2.8779378959181137) q[14];
rz(1.1585764731683383) q[14];
ry(2.3437594905186163) q[15];
rz(0.14611233014580363) q[15];
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
ry(-0.296973749339303) q[0];
rz(0.27326235152901285) q[0];
ry(1.4043680955999625) q[1];
rz(-2.7626866221943662) q[1];
ry(-1.7338723418460447) q[2];
rz(1.490070018272455) q[2];
ry(-1.7119393088013073) q[3];
rz(1.221673515440794) q[3];
ry(1.5249489476748586) q[4];
rz(-3.045629478713838) q[4];
ry(0.3598655471534955) q[5];
rz(2.243821032516783) q[5];
ry(-2.8820093208386837) q[6];
rz(3.0357293974240953) q[6];
ry(1.581576720626847) q[7];
rz(0.16779422945426764) q[7];
ry(3.084489072453951) q[8];
rz(-0.1125689762175206) q[8];
ry(3.0852686656480603) q[9];
rz(3.037797549470892) q[9];
ry(0.00030379665048136024) q[10];
rz(-3.0421765254533155) q[10];
ry(-3.140669403633786) q[11];
rz(-1.9042836661789089) q[11];
ry(-3.059813528902092) q[12];
rz(-0.450654789208983) q[12];
ry(-1.5682520376395697) q[13];
rz(-3.0560907628611154) q[13];
ry(1.3210758852169746) q[14];
rz(-1.0622618280733418) q[14];
ry(-1.9953492046881764) q[15];
rz(0.39866081961710575) q[15];
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
ry(-2.205461737483996) q[0];
rz(0.24550773021837813) q[0];
ry(-2.181781723663241) q[1];
rz(-2.806474796312755) q[1];
ry(2.8124743741288207) q[2];
rz(2.57576152302381) q[2];
ry(-0.1562541019919731) q[3];
rz(-2.0155188446479237) q[3];
ry(-0.002292283958134333) q[4];
rz(-1.641737645010461) q[4];
ry(-0.0008111890079976412) q[5];
rz(0.8829619163873322) q[5];
ry(3.1414510477495408) q[6];
rz(-1.5269152227809428) q[6];
ry(0.010079562150153267) q[7];
rz(-1.7982454871868754) q[7];
ry(-1.571755452819704) q[8];
rz(-2.574686038287861) q[8];
ry(1.571841366652572) q[9];
rz(3.120782699156672) q[9];
ry(-0.017605223488806665) q[10];
rz(-0.42200639995212363) q[10];
ry(-2.929479627056951) q[11];
rz(1.269988375611746) q[11];
ry(-1.5361926333851352) q[12];
rz(-0.4645547262282271) q[12];
ry(-1.2897518250300308) q[13];
rz(-0.4503883821984841) q[13];
ry(1.1266516085627218) q[14];
rz(2.6262119168877214) q[14];
ry(-0.2810544091195961) q[15];
rz(0.6693324529896629) q[15];
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
ry(-1.78747925675576) q[0];
rz(-1.2699332678666602) q[0];
ry(-1.688030432926583) q[1];
rz(0.006943313254671678) q[1];
ry(1.930256584027899) q[2];
rz(0.736561747542746) q[2];
ry(1.3658819695926692) q[3];
rz(-2.582689815606542) q[3];
ry(2.5789162692550187) q[4];
rz(-1.5659949922941914) q[4];
ry(3.014826330065764) q[5];
rz(-1.56137303566539) q[5];
ry(-0.017752457972438285) q[6];
rz(-1.7086070100239497) q[6];
ry(2.7998234976026435) q[7];
rz(-1.611348536637345) q[7];
ry(0.4424757814544765) q[8];
rz(-0.5752330353184619) q[8];
ry(1.5510027666877901) q[9];
rz(0.1058081108437845) q[9];
ry(1.475666008307406) q[10];
rz(-2.94055582517018) q[10];
ry(-2.7978283268610897) q[11];
rz(0.7224350247320411) q[11];
ry(1.9891830742868462) q[12];
rz(-1.9161605863441764) q[12];
ry(-2.1744709815152015) q[13];
rz(1.321324469178635) q[13];
ry(1.8987382624944757) q[14];
rz(-0.35977013275145087) q[14];
ry(1.4710550498033876) q[15];
rz(0.26522768124255336) q[15];
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
ry(2.7146700291708417) q[0];
rz(2.9303463319833827) q[0];
ry(-0.6329289928228745) q[1];
rz(1.697406610828689) q[1];
ry(0.3386314881587813) q[2];
rz(2.564359746620689) q[2];
ry(-0.11095564124640411) q[3];
rz(-1.0408759725526466) q[3];
ry(1.5831452695846595) q[4];
rz(3.1366239883799487) q[4];
ry(1.5591601641281034) q[5];
rz(3.1413282952662365) q[5];
ry(-3.079813922710736) q[6];
rz(2.9129245347090116) q[6];
ry(-0.022705496454371567) q[7];
rz(3.127592876939699) q[7];
ry(0.01442131965107662) q[8];
rz(-0.0481584026096774) q[8];
ry(-0.0006653152712328136) q[9];
rz(0.05240883104730756) q[9];
ry(-2.0642189021514495e-05) q[10];
rz(1.20303166379989) q[10];
ry(6.318066452415393e-05) q[11];
rz(2.034613016010634) q[11];
ry(0.0010706977215932305) q[12];
rz(2.850210329302605) q[12];
ry(0.0002738215844125749) q[13];
rz(1.1256956180578186) q[13];
ry(1.5394456619377808) q[14];
rz(0.0798611600187272) q[14];
ry(2.8893498332112864) q[15];
rz(-0.3772996238431672) q[15];
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
ry(0.47912294901077707) q[0];
rz(-2.3651039519899366) q[0];
ry(-1.1649885989326325) q[1];
rz(-1.7258519937239525) q[1];
ry(1.399845845533168) q[2];
rz(-3.1326093483629753) q[2];
ry(-2.0230280745319273) q[3];
rz(1.8834335632624712) q[3];
ry(1.5705474507591235) q[4];
rz(1.4964530708962642) q[4];
ry(-1.566377202265467) q[5];
rz(-0.29197625108540926) q[5];
ry(-1.1470136624487504) q[6];
rz(-2.4448083435445613) q[6];
ry(-1.5553979880018935) q[7];
rz(1.4382482525924412) q[7];
ry(-1.2463703948909899) q[8];
rz(1.5744047558019432) q[8];
ry(-3.1403069929877496) q[9];
rz(-1.9108201061351613) q[9];
ry(1.0768105600097941) q[10];
rz(-1.051440982203732) q[10];
ry(-2.469323097705522) q[11];
rz(-0.2269781294065443) q[11];
ry(-1.6357491940202247) q[12];
rz(-0.289221571313794) q[12];
ry(1.665549943702202) q[13];
rz(0.12535247112847525) q[13];
ry(2.9146364411484442) q[14];
rz(2.020446803548378) q[14];
ry(3.0186994069939184) q[15];
rz(2.0017127052239614) q[15];
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
ry(-1.6842448328773616) q[0];
rz(-3.03577373213922) q[0];
ry(0.8191440098853167) q[1];
rz(-1.8766248423310508) q[1];
ry(1.1915556664625617) q[2];
rz(2.4106776297161874) q[2];
ry(-1.2046657343088913) q[3];
rz(-2.786800053781749) q[3];
ry(-3.123703660344058) q[4];
rz(-0.3787501518177816) q[4];
ry(-3.1182160472874196) q[5];
rz(-0.49587311606463663) q[5];
ry(-2.766736649513562) q[6];
rz(0.43080477637788456) q[6];
ry(-3.1415886421715253) q[7];
rz(2.8361742418912987) q[7];
ry(-1.581286755976981) q[8];
rz(2.0225734849945294) q[8];
ry(-1.5796757765873382) q[9];
rz(1.5928453061569225) q[9];
ry(3.140675366868126) q[10];
rz(1.9210545612577725) q[10];
ry(0.014718901483918169) q[11];
rz(2.639522026022241) q[11];
ry(-0.005986870174903736) q[12];
rz(-0.7114923584288384) q[12];
ry(0.00964602754632418) q[13];
rz(-1.994132488232556) q[13];
ry(0.09412175100866893) q[14];
rz(-2.5448197182043564) q[14];
ry(-3.025182769062161) q[15];
rz(-0.8403739151496541) q[15];
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
ry(0.87862219024454) q[0];
rz(0.2777753590522157) q[0];
ry(0.8195353759788894) q[1];
rz(3.013010277625648) q[1];
ry(-0.24464227149340267) q[2];
rz(1.2727708531210027) q[2];
ry(-0.25711150442256775) q[3];
rz(0.30314787797358006) q[3];
ry(0.001391111245932386) q[4];
rz(3.065389400653796) q[4];
ry(3.140528137586034) q[5];
rz(1.9894327319321565) q[5];
ry(-3.139084704849114) q[6];
rz(2.9253110552451003) q[6];
ry(-0.0011112353129541994) q[7];
rz(1.7414354789732025) q[7];
ry(-0.018379842772601407) q[8];
rz(1.116180926707977) q[8];
ry(-0.28867458468694274) q[9];
rz(1.5479684788909744) q[9];
ry(3.105925727082178) q[10];
rz(1.0950011183377815) q[10];
ry(0.05682086492075644) q[11];
rz(-1.7971588996886376) q[11];
ry(-1.8600452934388283) q[12];
rz(-3.03238184682056) q[12];
ry(1.2344256138426044) q[13];
rz(0.15919482228769063) q[13];
ry(-1.8857182636029561) q[14];
rz(-2.7232260979255356) q[14];
ry(-0.48127025745509844) q[15];
rz(-0.8776374140797182) q[15];
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
ry(3.0922596290189657) q[0];
rz(1.272421267658986) q[0];
ry(1.4151637953500993) q[1];
rz(-2.565687654441181) q[1];
ry(-1.4909073070993415) q[2];
rz(-1.1279122776211352) q[2];
ry(0.536289904679113) q[3];
rz(1.1163235452705011) q[3];
ry(0.003335456613020149) q[4];
rz(-1.0325350041384076) q[4];
ry(3.1383783464612445) q[5];
rz(0.8466397229117647) q[5];
ry(1.7031162172361345) q[6];
rz(2.8520876090283718) q[6];
ry(-1.574474091769786) q[7];
rz(3.1400457426864246) q[7];
ry(1.5456236456944774) q[8];
rz(-0.11492772490138461) q[8];
ry(1.546845169369493) q[9];
rz(-2.1904104398641566) q[9];
ry(-0.00014868870598672146) q[10];
rz(1.8356842516167962) q[10];
ry(3.1410859767429202) q[11];
rz(0.24223400036544046) q[11];
ry(-1.4853162453335704) q[12];
rz(-0.5408805693799003) q[12];
ry(-1.659766709927049) q[13];
rz(1.56621941107024) q[13];
ry(-2.726571227290241) q[14];
rz(-0.9909090587979684) q[14];
ry(0.3665061523194124) q[15];
rz(-1.8244027199336468) q[15];
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
ry(3.0500054936759025) q[0];
rz(1.0078554952746335) q[0];
ry(2.0089818802070214) q[1];
rz(-2.674244265609558) q[1];
ry(-1.817572460486773) q[2];
rz(-1.3970853274727568) q[2];
ry(3.0012238565743243) q[3];
rz(1.7324327955774423) q[3];
ry(1.2150157161709303) q[4];
rz(2.4329851992733724) q[4];
ry(0.1144949331632672) q[5];
rz(-1.6279856440081621) q[5];
ry(1.5771071736591544) q[6];
rz(3.1388998917228044) q[6];
ry(-1.5710667834311245) q[7];
rz(3.1018591449977904) q[7];
ry(1.5909215276087492) q[8];
rz(0.4947496045920321) q[8];
ry(1.7286834800869313) q[9];
rz(-3.0982382178721) q[9];
ry(1.5657864303202977) q[10];
rz(1.414533144709604) q[10];
ry(0.0001015724481145952) q[11];
rz(-1.9758909866407646) q[11];
ry(-3.123153714852688) q[12];
rz(-2.1729872765560456) q[12];
ry(-2.873512724803524) q[13];
rz(0.004552593694719838) q[13];
ry(-2.186007463700531) q[14];
rz(1.3348831394415575) q[14];
ry(0.9848728946812773) q[15];
rz(1.2552188697554523) q[15];
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
ry(-3.099930488879943) q[0];
rz(0.005006195838770822) q[0];
ry(-0.3707625378265156) q[1];
rz(2.2670679370953435) q[1];
ry(0.000312998595913605) q[2];
rz(2.8938785062547554) q[2];
ry(-0.00040591746171142224) q[3];
rz(0.179862400716841) q[3];
ry(3.1414772656818157) q[4];
rz(-2.2877156167980535) q[4];
ry(-3.140662023408228) q[5];
rz(-0.025519942572340294) q[5];
ry(-1.5708543808295028) q[6];
rz(-3.136892668696179) q[6];
ry(1.572691371500377) q[7];
rz(3.1362533151202823) q[7];
ry(-0.0019775104232633103) q[8];
rz(-2.452715845221379) q[8];
ry(3.138184478084026) q[9];
rz(2.732306025260346) q[9];
ry(1.5712295407745431) q[10];
rz(-1.5543040369570678) q[10];
ry(-1.5706642682997671) q[11];
rz(1.3954414370513462) q[11];
ry(3.0078382349260413) q[12];
rz(-2.666647153214135) q[12];
ry(-0.8692454626871537) q[13];
rz(2.0846358670260283) q[13];
ry(-0.3156356775081846) q[14];
rz(1.6565789999849023) q[14];
ry(2.4553933069494502) q[15];
rz(1.3335138421542791) q[15];
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
ry(-2.0222183445625146) q[0];
rz(-1.5613420210462872) q[0];
ry(-0.6682254643141716) q[1];
rz(0.7403101281242236) q[1];
ry(1.7236850666447694) q[2];
rz(3.1136338351847077) q[2];
ry(-3.0318776052235057) q[3];
rz(2.1628447740387657) q[3];
ry(0.2415750524889555) q[4];
rz(-1.7028588196493075) q[4];
ry(-0.0879920370672842) q[5];
rz(-1.4754937923520564) q[5];
ry(1.5702062207276013) q[6];
rz(2.964644608543152) q[6];
ry(1.5888295644518358) q[7];
rz(1.463321100036702) q[7];
ry(-0.268874430734078) q[8];
rz(2.001728761711348) q[8];
ry(0.0019294623972179181) q[9];
rz(1.2728200062213186) q[9];
ry(-3.139391813387527) q[10];
rz(-0.030250813424900436) q[10];
ry(3.1408488518724007) q[11];
rz(2.646608562710197) q[11];
ry(-3.1413396984060054) q[12];
rz(2.0798422139933797) q[12];
ry(0.00028387985326538967) q[13];
rz(-0.5190877884927794) q[13];
ry(1.6202861320574693) q[14];
rz(-2.6098765367922034) q[14];
ry(-1.511602825288116) q[15];
rz(-2.761313484722486) q[15];
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
ry(1.3081056537158648) q[0];
rz(3.0781476744905576) q[0];
ry(1.551005047522022) q[1];
rz(-0.3477495892704141) q[1];
ry(-3.066518798208971) q[2];
rz(-1.3499366602824672) q[2];
ry(0.8661651691013885) q[3];
rz(1.5492536941978625) q[3];
ry(3.136874995364797) q[4];
rz(-0.028310769655982386) q[4];
ry(-3.1400997542024456) q[5];
rz(-0.933070230988414) q[5];
ry(-0.0015317472575063524) q[6];
rz(1.8778124117994839) q[6];
ry(-3.1400610589132114) q[7];
rz(1.7374122199281847) q[7];
ry(-0.0013715509548131166) q[8];
rz(1.1413490259242227) q[8];
ry(3.1408846685406395) q[9];
rz(0.6965470811712113) q[9];
ry(-0.012141609146138599) q[10];
rz(1.6176570504502261) q[10];
ry(-0.0003371375523633091) q[11];
rz(-1.2511909388725337) q[11];
ry(-1.5784743705568813) q[12];
rz(-0.07632336393305204) q[12];
ry(-1.5605268801505887) q[13];
rz(2.625687707329476) q[13];
ry(2.7495494977214974) q[14];
rz(2.1036101233690503) q[14];
ry(1.61812348706402) q[15];
rz(-0.14128494251320853) q[15];
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
ry(-1.5617626992683273) q[0];
rz(-1.5654963404213598) q[0];
ry(-1.5721827675622584) q[1];
rz(2.7336221458626255) q[1];
ry(1.5686039863950076) q[2];
rz(-0.007830412104854469) q[2];
ry(-1.5711720704084728) q[3];
rz(-0.01322718716062442) q[3];
ry(-0.0009133860285590895) q[4];
rz(0.36996890093560264) q[4];
ry(3.141524836438847) q[5];
rz(-0.6056237873221423) q[5];
ry(-3.136594932680113) q[6];
rz(-3.010335962071316) q[6];
ry(0.020269687924876413) q[7];
rz(1.0248730694690416) q[7];
ry(2.8714438585651116) q[8];
rz(-1.5847820113877442) q[8];
ry(-3.1397583220586616) q[9];
rz(2.6228741131644835) q[9];
ry(-1.5709546805200985) q[10];
rz(2.9000477222847634) q[10];
ry(1.5707548498431423) q[11];
rz(-1.666203929194582) q[11];
ry(0.0013644533429016277) q[12];
rz(0.9737658743183072) q[12];
ry(-0.0001872555258746189) q[13];
rz(-1.4473111994225365) q[13];
ry(1.3663785734441314) q[14];
rz(2.9857929040657742) q[14];
ry(-1.368804507190411) q[15];
rz(1.3744643870352518) q[15];
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
ry(0.4407739547019214) q[0];
rz(-1.5525216560536845) q[0];
ry(3.1382250519498855) q[1];
rz(2.521435690124003) q[1];
ry(-1.5789967594887822) q[2];
rz(-2.3020496468553433) q[2];
ry(-1.5720309132760881) q[3];
rz(3.0728155291107693) q[3];
ry(-0.00015074943444850822) q[4];
rz(2.663498823338946) q[4];
ry(3.140355201851818) q[5];
rz(0.4065486273218006) q[5];
ry(-1.590528829500197) q[6];
rz(1.5719039273231363) q[6];
ry(1.5725935614452442) q[7];
rz(-0.0022886437334464957) q[7];
ry(-3.1390640430342813) q[8];
rz(-1.032136624465715) q[8];
ry(0.04851306319244309) q[9];
rz(2.002964794333871) q[9];
ry(-1.494900586223988) q[10];
rz(-0.6768662058079971) q[10];
ry(2.9717740014532863) q[11];
rz(-1.7360889390885426) q[11];
ry(0.013477525750207286) q[12];
rz(-2.5976161446886854) q[12];
ry(-3.1339282386594727) q[13];
rz(0.45004366972531695) q[13];
ry(1.9522363017403501) q[14];
rz(-2.194533995519939) q[14];
ry(1.2385868454255002) q[15];
rz(0.09359216380882161) q[15];