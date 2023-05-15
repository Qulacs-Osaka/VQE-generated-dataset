OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.9166699096124473) q[0];
rz(-2.5483024617766437) q[0];
ry(-2.5632876676364234) q[1];
rz(-1.740790172857886) q[1];
ry(0.33936828615372594) q[2];
rz(-2.064099657739733) q[2];
ry(2.451555576532515) q[3];
rz(3.0027926554363913) q[3];
ry(-3.1258443421845747) q[4];
rz(-2.433045284933426) q[4];
ry(-2.609950881115286) q[5];
rz(-1.7241988829022459) q[5];
ry(-3.0323930514189956) q[6];
rz(1.9874055243046405) q[6];
ry(1.5749220166086593) q[7];
rz(2.7729763674368737) q[7];
ry(-1.569562927831497) q[8];
rz(1.5966571348939531) q[8];
ry(3.1354691669539965) q[9];
rz(-2.5124092423432334) q[9];
ry(-7.641719459439145e-05) q[10];
rz(1.1867089946585434) q[10];
ry(2.5653757117716225e-05) q[11];
rz(-2.702108671193631) q[11];
ry(-1.5541172059943247) q[12];
rz(1.7783551915429077) q[12];
ry(0.010365945841620892) q[13];
rz(-0.8909856382677397) q[13];
ry(-0.0013358120889902434) q[14];
rz(-1.6517923630348106) q[14];
ry(0.3624972625396909) q[15];
rz(2.3761276582237194) q[15];
ry(1.5778581386861248) q[16];
rz(1.5589746783516267) q[16];
ry(-1.5373800436150078) q[17];
rz(1.558043509241325) q[17];
ry(-0.002065398859895595) q[18];
rz(-1.4730149498715974) q[18];
ry(0.003410794214995836) q[19];
rz(2.8267151308873046) q[19];
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
ry(-2.576153592964733) q[0];
rz(-1.0058238626053575) q[0];
ry(1.4791431771374317) q[1];
rz(1.78292389320187) q[1];
ry(2.0257422009065307) q[2];
rz(-1.5815934564248957) q[2];
ry(-1.2183506626677854) q[3];
rz(-0.32924914404454375) q[3];
ry(-3.140127196813678) q[4];
rz(0.20641107287930985) q[4];
ry(-2.9723280853208447) q[5];
rz(2.8851452288517496) q[5];
ry(-0.014211010563406568) q[6];
rz(2.0340418646678913) q[6];
ry(-0.0028307174640005194) q[7];
rz(-1.060100931163974) q[7];
ry(1.4270721818450787) q[8];
rz(-2.832654407707408) q[8];
ry(-1.4996326090284144) q[9];
rz(2.7201643209518687) q[9];
ry(3.128989745489268) q[10];
rz(1.3413481981329296) q[10];
ry(-3.1410398006286857) q[11];
rz(2.9810937028735887) q[11];
ry(2.3550920283263976) q[12];
rz(0.1414935568239344) q[12];
ry(-1.3533246773862884) q[13];
rz(-1.5695100419970491) q[13];
ry(1.5572188318835674) q[14];
rz(-1.0043291096939366) q[14];
ry(1.5602698254175182) q[15];
rz(2.5498999104318845) q[15];
ry(-1.5712582337027596) q[16];
rz(1.4314086612179509) q[16];
ry(1.576234932546302) q[17];
rz(-0.6739280981339673) q[17];
ry(-1.2050646356057886) q[18];
rz(-0.12533599237276213) q[18];
ry(-2.503729151268078) q[19];
rz(1.0836638947731076) q[19];
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
ry(2.704956417126172) q[0];
rz(-1.3640417441005201) q[0];
ry(-1.5213986210820118) q[1];
rz(1.0660171752098506) q[1];
ry(1.0637800477433128) q[2];
rz(-2.2172978047017917) q[2];
ry(0.025548079827041232) q[3];
rz(1.9866903609192867) q[3];
ry(0.004347679433418961) q[4];
rz(0.08386996124825527) q[4];
ry(-2.5996943979617004) q[5];
rz(2.0888219128361616) q[5];
ry(0.022432864993066737) q[6];
rz(-2.6307545517285016) q[6];
ry(-0.01815600248335134) q[7];
rz(-0.5137615884886918) q[7];
ry(3.0855475353036383) q[8];
rz(-2.4737069952461677) q[8];
ry(-3.1398364662086724) q[9];
rz(1.01054833481695) q[9];
ry(-9.23803932900454e-05) q[10];
rz(0.29494546284807566) q[10];
ry(-0.02293804441773961) q[11];
rz(-1.0078974172138435) q[11];
ry(1.5688421531511683) q[12];
rz(-1.7415066446138752) q[12];
ry(-1.5684237747537628) q[13];
rz(-1.6897449952355976) q[13];
ry(0.004311755782300253) q[14];
rz(-1.2541944348895484) q[14];
ry(-2.9132173385853237) q[15];
rz(0.23816525348272943) q[15];
ry(3.12639762245138) q[16];
rz(1.3778053983042182) q[16];
ry(3.121037603438667) q[17];
rz(0.9494182771989947) q[17];
ry(-3.021218330311963) q[18];
rz(1.4661049920557787) q[18];
ry(-3.0485341715632552) q[19];
rz(-0.7146085347618891) q[19];
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
ry(1.9830447091840053) q[0];
rz(-0.8928936453901573) q[0];
ry(-2.0068502752689614) q[1];
rz(-1.4445457034542821) q[1];
ry(-0.7040317422301958) q[2];
rz(-0.02020968768845273) q[2];
ry(-1.473660087805992) q[3];
rz(1.6926609108115906) q[3];
ry(-0.002099211427998342) q[4];
rz(-0.3510787549257906) q[4];
ry(2.9068329783956193) q[5];
rz(1.839850249512894) q[5];
ry(3.129151986706209) q[6];
rz(-1.6849369673178825) q[6];
ry(0.0014722481952356146) q[7];
rz(-2.8154735019488255) q[7];
ry(3.075161489736978) q[8];
rz(-1.2426378688463204) q[8];
ry(-0.3552907111254413) q[9];
rz(-1.3632224828652264) q[9];
ry(1.5720147052097628) q[10];
rz(-1.5335506958591092) q[10];
ry(1.572286681174262) q[11];
rz(1.0105398024224295) q[11];
ry(-0.5323473824534798) q[12];
rz(-0.7182329301857234) q[12];
ry(3.00895751447641) q[13];
rz(2.7890827759904084) q[13];
ry(-1.5763853638471579) q[14];
rz(0.00015966113658373647) q[14];
ry(3.1408879236686422) q[15];
rz(0.5838872173579244) q[15];
ry(-1.5720520268056188) q[16];
rz(-1.0790336369643305) q[16];
ry(1.5579798849788657) q[17];
rz(2.5691883337894437) q[17];
ry(-1.3465907422469265) q[18];
rz(2.7229402218157657) q[18];
ry(-1.7129760996488448) q[19];
rz(-2.065057267083306) q[19];
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
ry(-2.715975397905702) q[0];
rz(-3.0988020160944703) q[0];
ry(-2.019878359132477) q[1];
rz(1.4416524341541173) q[1];
ry(0.3682900933329331) q[2];
rz(-0.03677056160135184) q[2];
ry(1.7415934366875474) q[3];
rz(-2.711916454102572) q[3];
ry(0.015669025926620783) q[4];
rz(-2.421113765588362) q[4];
ry(1.5703264472369565) q[5];
rz(0.015918132842067223) q[5];
ry(-1.5777157257162697) q[6];
rz(-2.7354228123550195) q[6];
ry(-1.5443419491103425) q[7];
rz(-1.488327928161547) q[7];
ry(-0.8763352038249266) q[8];
rz(2.5354208248589702) q[8];
ry(-1.5687350535916682) q[9];
rz(3.1388161912824857) q[9];
ry(2.5277605801542498) q[10];
rz(-0.4450726826933664) q[10];
ry(-3.127066717568777) q[11];
rz(-0.5521272582443507) q[11];
ry(-0.00019335298443579063) q[12];
rz(2.144653898134864) q[12];
ry(3.1405917671095804) q[13];
rz(-2.8227801202938183) q[13];
ry(-1.5645364819682177) q[14];
rz(-2.851245235696746) q[14];
ry(-0.00838666712359596) q[15];
rz(-1.113425884850104) q[15];
ry(-2.037226204146438) q[16];
rz(0.5238315300328855) q[16];
ry(-3.0996595227187878) q[17];
rz(-1.18473270503411) q[17];
ry(-3.007787709818985) q[18];
rz(2.8447775423928454) q[18];
ry(0.7663902999101724) q[19];
rz(1.0312263407972377) q[19];
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
ry(-1.990426056109845) q[0];
rz(-0.09652573503030926) q[0];
ry(-2.7439784923200516) q[1];
rz(-2.9097084118759797) q[1];
ry(-1.7912003357044548) q[2];
rz(0.6172658261467813) q[2];
ry(-2.72661841726496) q[3];
rz(1.5895965872809337) q[3];
ry(3.135973537799288) q[4];
rz(1.8846190311793611) q[4];
ry(-5.659674198721376e-05) q[5];
rz(-1.0540758500581688) q[5];
ry(0.0004982596772167233) q[6];
rz(-1.9518169349548433) q[6];
ry(0.022907246040335675) q[7];
rz(-0.864224795003308) q[7];
ry(0.013789726373978847) q[8];
rz(1.8847723055983312) q[8];
ry(1.5815216598689275) q[9];
rz(-3.140972771827496) q[9];
ry(-1.5781856901329192) q[10];
rz(2.2873914863530076) q[10];
ry(-1.5744168311388709) q[11];
rz(-1.6811230178099192) q[11];
ry(-0.7291584612004077) q[12];
rz(-1.3788749451504723) q[12];
ry(2.3412642238049455) q[13];
rz(1.438009521320181) q[13];
ry(-3.141367797000394) q[14];
rz(2.3817515892596743) q[14];
ry(0.00024397314417967664) q[15];
rz(2.6265797702003524) q[15];
ry(3.1111529337760975) q[16];
rz(-0.5588155374298953) q[16];
ry(-0.11690418200535031) q[17];
rz(-3.1164384148435453) q[17];
ry(0.026736106521110692) q[18];
rz(0.18245220550700098) q[18];
ry(-0.024105318275225857) q[19];
rz(0.41748476831709214) q[19];
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
ry(-0.35283424403326347) q[0];
rz(1.3076142999950366) q[0];
ry(-1.9710359367991062) q[1];
rz(0.7217972289628058) q[1];
ry(0.5998829354976465) q[2];
rz(2.6828453384824695) q[2];
ry(1.155825269899647) q[3];
rz(0.8747245531464088) q[3];
ry(3.141016503939834) q[4];
rz(-0.5088523352015959) q[4];
ry(0.370736612417299) q[5];
rz(0.34168855071042703) q[5];
ry(1.5724043803075638) q[6];
rz(-3.1389066961997467) q[6];
ry(3.1399971873517525) q[7];
rz(0.8006760573971725) q[7];
ry(-3.115414335934416) q[8];
rz(-0.3515810235803216) q[8];
ry(-1.5714202356165223) q[9];
rz(-3.077306559279595) q[9];
ry(-0.0076898469865853795) q[10];
rz(0.2351675947179022) q[10];
ry(-3.115614897199287) q[11];
rz(-0.04658638208906905) q[11];
ry(0.0431844906572616) q[12];
rz(1.3865070122408942) q[12];
ry(-3.087921163685669) q[13];
rz(-1.7128164687077845) q[13];
ry(-3.1400811266447426) q[14];
rz(-0.15984316851706826) q[14];
ry(0.0007410007629387039) q[15];
rz(-2.8038435375454003) q[15];
ry(-0.5645883402783323) q[16];
rz(-0.5411287480849971) q[16];
ry(-0.11917572585972991) q[17];
rz(-0.09903780957126904) q[17];
ry(1.4152970376238678) q[18];
rz(2.585849880920286) q[18];
ry(-2.765729271113699) q[19];
rz(-1.8923568939150766) q[19];
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
ry(-0.6693531338034671) q[0];
rz(2.7611925374416884) q[0];
ry(-2.4140424682949178) q[1];
rz(-1.2227883689738839) q[1];
ry(-1.268744617879869) q[2];
rz(1.2631219879605418) q[2];
ry(1.3657039993836326) q[3];
rz(-3.0920934305746366) q[3];
ry(-3.1399856353541695) q[4];
rz(-2.8823732445095738) q[4];
ry(3.139650769476786) q[5];
rz(0.5165402812846552) q[5];
ry(1.5712809694920855) q[6];
rz(-1.5515353485382581) q[6];
ry(-1.570567399235509) q[7];
rz(-1.5423259323837213) q[7];
ry(1.5844156325423902) q[8];
rz(2.412279903750948) q[8];
ry(-0.037536360588375395) q[9];
rz(2.602565394777134) q[9];
ry(-3.138892452819589) q[10];
rz(-2.179586037088713) q[10];
ry(-3.1413772443170767) q[11];
rz(0.05985035378674119) q[11];
ry(2.389807141903032) q[12];
rz(-1.684564929930369) q[12];
ry(2.3420417348860965) q[13];
rz(1.478092264867494) q[13];
ry(-3.1395286329513086) q[14];
rz(-0.6832414297304802) q[14];
ry(-0.032906713286184684) q[15];
rz(-1.6202744835419969) q[15];
ry(1.5212913320731458) q[16];
rz(-1.6819113964499763) q[16];
ry(-0.04081010408509317) q[17];
rz(2.308193076027329) q[17];
ry(1.5166938394698875) q[18];
rz(0.12357842104670878) q[18];
ry(-0.007916251558865106) q[19];
rz(0.4626630403414805) q[19];
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
ry(-0.9463950604469186) q[0];
rz(0.9730217514618824) q[0];
ry(0.08912423495559257) q[1];
rz(0.22146437683526177) q[1];
ry(-1.2988034046796006) q[2];
rz(1.0546062642919807) q[2];
ry(-1.9971454654751661) q[3];
rz(1.1550343845559368) q[3];
ry(-1.5731453262248347) q[4];
rz(-1.970017552645776) q[4];
ry(1.0276726646770544) q[5];
rz(2.6230863644467313) q[5];
ry(2.7153218911295514) q[6];
rz(0.35877570412305015) q[6];
ry(1.4915378763144032) q[7];
rz(0.32761968706245903) q[7];
ry(-3.1360048811644883) q[8];
rz(1.4795101854106) q[8];
ry(-1.572460893194087) q[9];
rz(1.8518785199948367) q[9];
ry(1.5675006604223716) q[10];
rz(3.078888804999069) q[10];
ry(-1.5762153108299886) q[11];
rz(1.5786798611153818) q[11];
ry(0.05357938011417136) q[12];
rz(3.123008250300728) q[12];
ry(-3.092752741609806) q[13];
rz(0.8670197188610959) q[13];
ry(1.5709964169059993) q[14];
rz(2.634064104245382) q[14];
ry(-1.5721664389199936) q[15];
rz(0.4684625506047331) q[15];
ry(-3.070739200801484) q[16];
rz(3.0303331082607143) q[16];
ry(-3.105994001879913) q[17];
rz(1.789915846202353) q[17];
ry(0.8937878049995515) q[18];
rz(-0.07697445839349729) q[18];
ry(-1.5864087117316057) q[19];
rz(0.009826654789008096) q[19];
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
ry(0.6170273068013872) q[0];
rz(-0.09376081296474217) q[0];
ry(0.055415855395339435) q[1];
rz(-2.8041135700189495) q[1];
ry(-1.5805870149076546) q[2];
rz(-1.3776076660966052) q[2];
ry(-1.6010009726147938) q[3];
rz(-1.5712399795756697) q[3];
ry(-3.140904976439388) q[4];
rz(2.746013039694862) q[4];
ry(-0.00034400116323891894) q[5];
rz(-2.567920470007388) q[5];
ry(-0.0010066746820145411) q[6];
rz(2.5989466890122546) q[6];
ry(0.006724296578696887) q[7];
rz(2.816225324577355) q[7];
ry(-3.1400742127959465) q[8];
rz(-2.5015989082210828) q[8];
ry(3.092965568633694) q[9];
rz(-2.8575539801794396) q[9];
ry(1.5858860205516327) q[10];
rz(-0.7719178666767608) q[10];
ry(1.5656623002893753) q[11];
rz(0.9770585948051628) q[11];
ry(0.6038516485404735) q[12];
rz(-3.139627302961564) q[12];
ry(-3.1290643155124536) q[13];
rz(3.027147830481667) q[13];
ry(3.1372908902196555) q[14];
rz(-0.5069551241501227) q[14];
ry(3.141093303054146) q[15];
rz(0.5065872051105799) q[15];
ry(3.0904511792491247) q[16];
rz(-0.37812426703223645) q[16];
ry(-0.0002689828062436561) q[17];
rz(-2.165605339250231) q[17];
ry(1.5928439394303808) q[18];
rz(-3.1350604762165526) q[18];
ry(1.5800690081334703) q[19];
rz(-1.058522042490891) q[19];
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
ry(-1.4305609896135878) q[0];
rz(2.649711892594316) q[0];
ry(2.447203977684239) q[1];
rz(1.8170647953782526) q[1];
ry(0.05826926438055491) q[2];
rz(-0.27653847406061244) q[2];
ry(2.358314317387308) q[3];
rz(-1.502767108194476) q[3];
ry(2.9428065115646347) q[4];
rz(-1.5794845024256956) q[4];
ry(3.137745978830157) q[5];
rz(0.0528363222013008) q[5];
ry(-3.103970153805315) q[6];
rz(-0.1004386589418685) q[6];
ry(-1.4760728523085387) q[7];
rz(1.2751934818850028) q[7];
ry(-1.5739697442600784) q[8];
rz(-3.126954312558186) q[8];
ry(2.37385873982357) q[9];
rz(-1.5683270366482056) q[9];
ry(0.002117929106255181) q[10];
rz(2.3551425254527394) q[10];
ry(0.0025370027584129214) q[11];
rz(-0.9658632485487919) q[11];
ry(-2.4426471743408813) q[12];
rz(2.313234330452215) q[12];
ry(2.952493178888687) q[13];
rz(0.11529212596075168) q[13];
ry(-1.57443936755127) q[14];
rz(-1.2164778450863332) q[14];
ry(1.5654558359029438) q[15];
rz(-1.6955841893064976) q[15];
ry(-3.079167808664522) q[16];
rz(2.799470957281019) q[16];
ry(-3.013123584779925) q[17];
rz(-0.4256825838853091) q[17];
ry(-1.5651418068338703) q[18];
rz(-1.4380324144021932) q[18];
ry(-3.0302048086192377) q[19];
rz(-0.4284277504104885) q[19];
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
ry(2.0273845419208874) q[0];
rz(-1.5943177341316095) q[0];
ry(0.7196718740664735) q[1];
rz(-1.7510189613450597) q[1];
ry(-2.644080949384662) q[2];
rz(-0.050813797794132746) q[2];
ry(1.6306880805066608) q[3];
rz(-0.14987638027844244) q[3];
ry(-1.5919834017279744) q[4];
rz(1.641461994316582) q[4];
ry(1.5783418010817902) q[5];
rz(1.571062034176948) q[5];
ry(-3.1409712330083748) q[6];
rz(-1.4697644497269735) q[6];
ry(0.0007176647756770294) q[7];
rz(-1.4455043116414217) q[7];
ry(-1.5685878809369311) q[8];
rz(1.5776326578443358) q[8];
ry(-1.5706907670645125) q[9];
rz(-1.5683308259667348) q[9];
ry(-1.5721378858273924) q[10];
rz(0.24681380897930377) q[10];
ry(-1.5716744989992368) q[11];
rz(1.8223844678718215) q[11];
ry(2.9105534654715326) q[12];
rz(-0.8236248812977761) q[12];
ry(0.018871900641286743) q[13];
rz(1.2732987550634016) q[13];
ry(-3.1400947165805158) q[14];
rz(-0.5855129171834713) q[14];
ry(3.139932107165776) q[15];
rz(1.3552774155410479) q[15];
ry(1.58431647993772) q[16];
rz(2.765919751614886) q[16];
ry(1.5667070185288845) q[17];
rz(-1.8221163416363861) q[17];
ry(0.5259263142441197) q[18];
rz(-1.6758135404384653) q[18];
ry(-0.14880365249206928) q[19];
rz(2.1352385705711616) q[19];
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
ry(0.004929403885761597) q[0];
rz(-2.905877634276914) q[0];
ry(-1.5378020389182439) q[1];
rz(2.7800651628244015) q[1];
ry(1.6215217159276651) q[2];
rz(2.621751923380466) q[2];
ry(-1.4603179242150688) q[3];
rz(1.602022920330711) q[3];
ry(-3.1335689852077464) q[4];
rz(2.4488115217628303) q[4];
ry(-1.565840502251817) q[5];
rz(0.14654637416411112) q[5];
ry(1.5694914494988441) q[6];
rz(-3.0297454157826906) q[6];
ry(0.0016949841568738575) q[7];
rz(1.7120846730784507) q[7];
ry(-1.5710663127148963) q[8];
rz(3.14045201078386) q[8];
ry(1.5710358354903171) q[9];
rz(1.5741119959636511) q[9];
ry(-0.005181632388863472) q[10];
rz(2.8907923023089) q[10];
ry(-3.140225458757185) q[11];
rz(1.8220357924133403) q[11];
ry(0.005972423848290553) q[12];
rz(-2.009543053546867) q[12];
ry(-0.00081777093545643) q[13];
rz(1.1207951225920647) q[13];
ry(3.1413565231708125) q[14];
rz(2.7465007197030284) q[14];
ry(-3.1377310413630304) q[15];
rz(-0.41254583401492256) q[15];
ry(0.044270690632201855) q[16];
rz(-1.069220350012201) q[16];
ry(-1.5367771155015255) q[17];
rz(1.4102432169157495) q[17];
ry(-3.1225356421884505) q[18];
rz(-1.5952415972484602) q[18];
ry(0.004621283150501476) q[19];
rz(0.9595143841052841) q[19];
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
ry(1.5676903529252018) q[0];
rz(0.7613839693359463) q[0];
ry(3.1374811318849005) q[1];
rz(1.218558913693217) q[1];
ry(3.1171692617189772) q[2];
rz(-2.103027844149035) q[2];
ry(-1.5648297803665532) q[3];
rz(-1.2650954349144266) q[3];
ry(3.1414116900951714) q[4];
rz(2.532885521766065) q[4];
ry(-3.1410232938857368) q[5];
rz(0.20196301307584455) q[5];
ry(-3.1393512176071106) q[6];
rz(-3.0307821049279005) q[6];
ry(1.057060619212892) q[7];
rz(1.7981061713774023) q[7];
ry(-1.5714346961641485) q[8];
rz(-2.9340883392999917) q[8];
ry(2.6854999791068446) q[9];
rz(-2.2298933684967226) q[9];
ry(-1.1942909080838158) q[10];
rz(-1.8040387393495427) q[10];
ry(1.5534967578242627) q[11];
rz(-2.4305876421138204) q[11];
ry(-2.7488885037031743) q[12];
rz(1.1375155324349153) q[12];
ry(3.1153414861540965) q[13];
rz(-0.7583472991133604) q[13];
ry(-0.00015613018302175873) q[14];
rz(0.9612844077007117) q[14];
ry(-0.0007523326593088619) q[15];
rz(2.730292964456089) q[15];
ry(1.7436301923484463) q[16];
rz(-1.367383863032222) q[16];
ry(0.025100812753580115) q[17];
rz(0.163259330362527) q[17];
ry(1.5633273487065436) q[18];
rz(-1.0096468023875822) q[18];
ry(0.00913774822393787) q[19];
rz(2.583671129704048) q[19];
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
ry(0.18799153429288293) q[0];
rz(0.8409788555682027) q[0];
ry(-2.7516703783357555) q[1];
rz(-3.0180394215617246) q[1];
ry(1.6560753352916067) q[2];
rz(-1.5065819564014147) q[2];
ry(-0.015496175683504835) q[3];
rz(-1.8919871901939869) q[3];
ry(0.001558128261103242) q[4];
rz(2.108918563954666) q[4];
ry(-0.5946832719572815) q[5];
rz(0.13465327127665178) q[5];
ry(1.5526598226217612) q[6];
rz(1.5690025157271146) q[6];
ry(3.1412184955350706) q[7];
rz(0.22643742323014052) q[7];
ry(3.140383490601144) q[8];
rz(0.2055776481039201) q[8];
ry(-3.1410686738778555) q[9];
rz(-0.6525323522159262) q[9];
ry(4.0710295298943076e-05) q[10];
rz(1.7973346255217204) q[10];
ry(3.141489046683501) q[11];
rz(0.6924174072173344) q[11];
ry(-1.5375898340133893) q[12];
rz(1.6133306111255197) q[12];
ry(0.09240494076223259) q[13];
rz(1.6872334429268825) q[13];
ry(-0.0021146880582603536) q[14];
rz(-1.5482565145158902) q[14];
ry(3.1261006245897454) q[15];
rz(-0.7459022170623149) q[15];
ry(-1.6704007778686256) q[16];
rz(-2.529204828639882) q[16];
ry(1.5690297980899377) q[17];
rz(2.9999662937429705) q[17];
ry(3.1379769385510863) q[18];
rz(-2.582134770647463) q[18];
ry(0.3018833528848859) q[19];
rz(2.8440386021770756) q[19];
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
ry(-1.598682342945846) q[0];
rz(1.4249880411058315) q[0];
ry(-1.5435840975654678) q[1];
rz(-3.1280520074687495) q[1];
ry(0.0033488283236504652) q[2];
rz(1.5167588001553804) q[2];
ry(-1.5674970186575914) q[3];
rz(1.5829126945299383) q[3];
ry(0.00020500224340364988) q[4];
rz(2.5970957184419268) q[4];
ry(-0.0016718112630309536) q[5];
rz(-0.1522645188749392) q[5];
ry(-1.5710180214462641) q[6];
rz(3.1413785778187537) q[6];
ry(-1.5698306009029777) q[7];
rz(1.415203038704948) q[7];
ry(-1.5699484521909863) q[8];
rz(-0.0002961332068842951) q[8];
ry(1.5712697100056943) q[9];
rz(1.031254284567515) q[9];
ry(-2.6899742150100834) q[10];
rz(-1.5787646207260195) q[10];
ry(0.018956102944522435) q[11];
rz(2.5289607302698487) q[11];
ry(0.21094915756385238) q[12];
rz(1.538751873797188) q[12];
ry(-3.1133628567152134) q[13];
rz(1.5113153297728923) q[13];
ry(-0.016848199535245634) q[14];
rz(1.6216197888908779) q[14];
ry(0.11597636639926989) q[15];
rz(-2.9214296484092483) q[15];
ry(-3.13273589153251) q[16];
rz(-2.100508706562573) q[16];
ry(1.9316315367986032) q[17];
rz(1.548694827918214) q[17];
ry(-1.6495457228577406) q[18];
rz(-1.2331015196329682) q[18];
ry(-1.6113718367044376) q[19];
rz(-0.05166088788201405) q[19];
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
ry(1.5765117706490717) q[0];
rz(-1.4468298171001541) q[0];
ry(1.5712700057042284) q[1];
rz(2.948514602673094) q[1];
ry(-1.5755307818293858) q[2];
rz(-3.013999510809649) q[2];
ry(-1.5779987534780755) q[3];
rz(-2.7036633056760486) q[3];
ry(-0.0010427943552890895) q[4];
rz(1.554257731791573) q[4];
ry(-1.546609447237447) q[5];
rz(-0.781208712202142) q[5];
ry(1.5715023764379419) q[6];
rz(0.11310439062310079) q[6];
ry(-3.1408079903427857) q[7];
rz(-1.92255191377913) q[7];
ry(-1.5704715472014814) q[8];
rz(1.6959330875081866) q[8];
ry(-3.141535876028633) q[9];
rz(-2.274182958534687) q[9];
ry(-1.5661127226951723) q[10];
rz(-3.015757705416332) q[10];
ry(3.1402724089555956) q[11];
rz(-2.3771429439323883) q[11];
ry(2.4269526323433332) q[12];
rz(-3.0078896997518405) q[12];
ry(-1.600341930091333) q[13];
rz(2.970449898687751) q[13];
ry(-1.5790497503982976) q[14];
rz(1.68661126939195) q[14];
ry(3.1407445240944694) q[15];
rz(-3.0989337361981426) q[15];
ry(3.1386193787076366) q[16];
rz(-2.5380743812523074) q[16];
ry(-1.6142461009076663) q[17];
rz(1.3792896692662335) q[17];
ry(0.022884096749765794) q[18];
rz(2.9167731789466473) q[18];
ry(0.6810151396006026) q[19];
rz(-0.15536255097742924) q[19];