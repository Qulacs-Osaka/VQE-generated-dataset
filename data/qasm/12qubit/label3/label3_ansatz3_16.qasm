OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.7720339978571387) q[0];
rz(-0.8381444145068706) q[0];
ry(-2.085115543799424) q[1];
rz(1.1291869549653044) q[1];
ry(-1.3961636038037815) q[2];
rz(-2.5705778141224243) q[2];
ry(1.5635495149162588) q[3];
rz(-2.91201859466776) q[3];
ry(-0.0018008705647429705) q[4];
rz(0.8383123928539978) q[4];
ry(0.0005715918135784648) q[5];
rz(-2.8060340410775466) q[5];
ry(-1.5367126303266225) q[6];
rz(0.73144073549995) q[6];
ry(1.8910637708521598) q[7];
rz(-1.4011208274144231) q[7];
ry(-2.884478104115011) q[8];
rz(0.23236429484541826) q[8];
ry(-1.2329639712704625) q[9];
rz(-2.34264483717726) q[9];
ry(-1.3225814911682017) q[10];
rz(2.510419554065988) q[10];
ry(-1.547856609106457) q[11];
rz(3.1127214844632696) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.2551715055226467) q[0];
rz(-2.166416236274511) q[0];
ry(0.13964997191344858) q[1];
rz(-1.2406370693986437) q[1];
ry(0.007262403508514125) q[2];
rz(-2.1100454245346243) q[2];
ry(-1.557759490995199) q[3];
rz(2.7960530983976493) q[3];
ry(-1.5439494164257663) q[4];
rz(-3.131005452903852) q[4];
ry(3.141177239083062) q[5];
rz(-2.5929696788130303) q[5];
ry(-1.1050022783248217) q[6];
rz(-1.7992853874303947) q[6];
ry(1.9957223623041804) q[7];
rz(2.3182708296772465) q[7];
ry(-1.5370552453549902) q[8];
rz(1.5673692621747692) q[8];
ry(-1.0730684912440553) q[9];
rz(2.136589390514878) q[9];
ry(-1.0769724938612268) q[10];
rz(2.338905928965593) q[10];
ry(0.038434000573814284) q[11];
rz(-3.0589762286597235) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.7747922805981686) q[0];
rz(-1.159721607901132) q[0];
ry(1.6695433829903799) q[1];
rz(1.9807899828439437) q[1];
ry(1.6008352885541315) q[2];
rz(1.2620100960136715) q[2];
ry(1.6210080867047179) q[3];
rz(-0.10207090687231446) q[3];
ry(-0.0018195583457654152) q[4];
rz(-0.47280988910479643) q[4];
ry(0.00042978868355674675) q[5];
rz(-2.6292434871601573) q[5];
ry(-0.003195437394863948) q[6];
rz(-0.21088341799483332) q[6];
ry(0.4270186059433625) q[7];
rz(1.435142760633866) q[7];
ry(-0.8949877820169447) q[8];
rz(1.2770803112364595) q[8];
ry(-1.3113739258994288) q[9];
rz(-1.9924776010166951) q[9];
ry(1.610455796319267) q[10];
rz(-0.7645789385736599) q[10];
ry(0.2582348200465425) q[11];
rz(1.576870863772363) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.5819226546633096) q[0];
rz(0.03883639881416645) q[0];
ry(0.24477319250742188) q[1];
rz(1.9874430599786583) q[1];
ry(1.6031020536173333) q[2];
rz(1.8531098537524986) q[2];
ry(-2.7238928373395366) q[3];
rz(0.19045760550001994) q[3];
ry(-2.5176306892488634) q[4];
rz(-3.0210186846437495) q[4];
ry(-3.1414275579598536) q[5];
rz(-0.024155985037463707) q[5];
ry(0.4619837778956907) q[6];
rz(-0.3499384941482857) q[6];
ry(0.5660271273579323) q[7];
rz(0.5482449478551119) q[7];
ry(2.7419339928769664) q[8];
rz(-1.616947528499952) q[8];
ry(1.0665964080104156) q[9];
rz(-2.7779325913123305) q[9];
ry(-2.633043061904922) q[10];
rz(-2.81498543949902) q[10];
ry(-2.4329766728529107) q[11];
rz(-1.3185159799614752) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.9867869557790994) q[0];
rz(1.3048955452016506) q[0];
ry(-2.9973277579056) q[1];
rz(-2.4511243153957856) q[1];
ry(-1.558674500844794) q[2];
rz(-0.30917158072626927) q[2];
ry(2.3637352389584887) q[3];
rz(-0.7219664577392999) q[3];
ry(-3.138298780356135) q[4];
rz(2.4786032416318005) q[4];
ry(-0.007227918591663496) q[5];
rz(3.0738246083408813) q[5];
ry(-3.139197136735484) q[6];
rz(2.512440985197324) q[6];
ry(0.9501292424956519) q[7];
rz(2.4740722310092362) q[7];
ry(0.6383882771378495) q[8];
rz(-1.6237625199424777) q[8];
ry(1.9032233401959682) q[9];
rz(-1.1290803159660963) q[9];
ry(2.287067738183787) q[10];
rz(-1.495623983259705) q[10];
ry(-0.525158407323131) q[11];
rz(2.297041732288898) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.921635403350125) q[0];
rz(3.0299866919713416) q[0];
ry(-0.8452315601261038) q[1];
rz(2.066339515466728) q[1];
ry(2.5363954589900635) q[2];
rz(-1.5976613014632977) q[2];
ry(-0.3614389013964187) q[3];
rz(3.051970060114412) q[3];
ry(-1.5453520058058903) q[4];
rz(-1.2490117371166916) q[4];
ry(0.0001516929380072085) q[5];
rz(2.050717112600551) q[5];
ry(0.174573735381707) q[6];
rz(1.871417357845609) q[6];
ry(2.6184688523790483) q[7];
rz(0.19525677628173096) q[7];
ry(-1.4377467372872461) q[8];
rz(1.2010047682013294) q[8];
ry(-1.2716204098803703) q[9];
rz(-2.2317302695142915) q[9];
ry(-0.785037395865131) q[10];
rz(1.4098169788893147) q[10];
ry(-2.143468538506103) q[11];
rz(1.2506435168317829) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.1019836222252613) q[0];
rz(-1.4288730184453824) q[0];
ry(-1.0119452558845456) q[1];
rz(-2.5761740811720926) q[1];
ry(1.6185448302438843) q[2];
rz(1.353471441842503) q[2];
ry(1.498861297785201) q[3];
rz(1.9527121805458172) q[3];
ry(-1.5663890092317256) q[4];
rz(-0.003215636293630197) q[4];
ry(-3.1305496932319197) q[5];
rz(1.0236472084579173) q[5];
ry(-3.136288066011167) q[6];
rz(-2.434122472749015) q[6];
ry(-1.252760432140967) q[7];
rz(0.07520826248922853) q[7];
ry(-1.0541129856731437) q[8];
rz(3.126568888179468) q[8];
ry(-1.1495054812554129) q[9];
rz(-3.0558550588390787) q[9];
ry(1.3579499400233963) q[10];
rz(-0.18385021417404507) q[10];
ry(-2.4885172525778834) q[11];
rz(-1.0377299620535754) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.622113841008077) q[0];
rz(-2.2762414523872265) q[0];
ry(0.7578761597120235) q[1];
rz(1.5178000628092816) q[1];
ry(-1.211407644489391) q[2];
rz(1.867273480607153) q[2];
ry(2.9466975558845503) q[3];
rz(1.0362606741418658) q[3];
ry(-1.5123592532781163) q[4];
rz(0.015577316718074385) q[4];
ry(-3.1377187636767) q[5];
rz(0.49300769577013437) q[5];
ry(-0.06442948753181099) q[6];
rz(-0.9315469340734402) q[6];
ry(0.01799018133005337) q[7];
rz(-0.011074931742688145) q[7];
ry(-1.0705371717520942) q[8];
rz(-2.29510438566341) q[8];
ry(-2.3921369223983673) q[9];
rz(0.3963523879472665) q[9];
ry(1.3443831127597985) q[10];
rz(0.8998986272544683) q[10];
ry(1.4973376648238652) q[11];
rz(2.3700169953678016) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.1390127211640304) q[0];
rz(0.619703094624409) q[0];
ry(2.2893029870710744) q[1];
rz(-0.2734224301676394) q[1];
ry(-3.138697912686512) q[2];
rz(-2.9993789944441627) q[2];
ry(0.002268194589437833) q[3];
rz(-0.9260368795685981) q[3];
ry(0.04428737764636548) q[4];
rz(2.804081461154196) q[4];
ry(2.7005720506850186) q[5];
rz(-0.928334980734701) q[5];
ry(3.1272684694533814) q[6];
rz(0.8221080081274169) q[6];
ry(2.9872707994800707) q[7];
rz(0.1734136052727699) q[7];
ry(-1.3871835960026127) q[8];
rz(-0.06544936850640212) q[8];
ry(-1.471720161546678) q[9];
rz(-0.37913229630676337) q[9];
ry(-1.4758094523703367) q[10];
rz(1.0985579737715188) q[10];
ry(1.2492182674436145) q[11];
rz(-0.7052354040161264) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.5267378017965489) q[0];
rz(-2.202093927418676) q[0];
ry(-1.298677032888162) q[1];
rz(-0.09897714200729536) q[1];
ry(1.1788263707069664) q[2];
rz(0.392609221246728) q[2];
ry(1.4847636231662218) q[3];
rz(0.0027880199729350963) q[3];
ry(-3.136827356843109) q[4];
rz(2.8995989059691443) q[4];
ry(-0.0011788218234318748) q[5];
rz(-2.2964413203547) q[5];
ry(0.3550618562403711) q[6];
rz(1.3478350656166693) q[6];
ry(-3.1404262142291923) q[7];
rz(-2.0030685300099083) q[7];
ry(0.5237898269048277) q[8];
rz(1.1181982666026737) q[8];
ry(-1.560179203646227) q[9];
rz(-1.3857204485532582) q[9];
ry(-0.08410468826682482) q[10];
rz(-1.589501458176306) q[10];
ry(-1.2956381280494709) q[11];
rz(1.0792328637585864) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.6085842480863999) q[0];
rz(0.2691480059607585) q[0];
ry(-2.141748779689535) q[1];
rz(-3.0931010810352757) q[1];
ry(1.5737094493393742) q[2];
rz(0.45476858050003166) q[2];
ry(-1.5738085002281812) q[3];
rz(3.1409209926436485) q[3];
ry(-3.1371128244572253) q[4];
rz(1.0623077992506333) q[4];
ry(-1.605334718812916) q[5];
rz(-2.3979768737674245) q[5];
ry(3.129916473352492) q[6];
rz(-0.21485103517376927) q[6];
ry(2.2077515529636593) q[7];
rz(3.064463421049499) q[7];
ry(-3.1190736443590974) q[8];
rz(-1.696593202212996) q[8];
ry(-2.0469401312492255) q[9];
rz(-0.6186926178236166) q[9];
ry(1.5405065989326747) q[10];
rz(-0.9766421771676798) q[10];
ry(-1.7215144196288596) q[11];
rz(-1.8346395021711963) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.0015095998238657154) q[0];
rz(-1.814881129724741) q[0];
ry(-2.185536434153467) q[1];
rz(-3.0671834234589053) q[1];
ry(0.13366382855677908) q[2];
rz(2.666116187637451) q[2];
ry(1.589247666192697) q[3];
rz(-0.11944733760522741) q[3];
ry(3.1363898664287944) q[4];
rz(1.285258288052784) q[4];
ry(0.024542718102514627) q[5];
rz(-2.840466665597259) q[5];
ry(1.8394480728360723) q[6];
rz(1.1203207469105347) q[6];
ry(-0.00010748325247345036) q[7];
rz(-1.8694398613860879) q[7];
ry(-0.13457489249047594) q[8];
rz(-0.5701485120614086) q[8];
ry(3.1038696638027976) q[9];
rz(1.2719059584936494) q[9];
ry(-1.4000852853158037) q[10];
rz(-1.370582718838877) q[10];
ry(1.222600381224212) q[11];
rz(3.0184130419693838) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.421339921134158) q[0];
rz(-3.115543793007331) q[0];
ry(-1.579749372174117) q[1];
rz(-0.999614649305586) q[1];
ry(1.8789681932116782) q[2];
rz(-3.0739518725384523) q[2];
ry(-3.137739286842403) q[3];
rz(-0.08847608388439543) q[3];
ry(-0.00044179690711844736) q[4];
rz(1.3461801903642723) q[4];
ry(-0.001152550681228881) q[5];
rz(-1.0528686971752563) q[5];
ry(0.008414153354893018) q[6];
rz(-1.3453519171677062) q[6];
ry(3.1035669808243216) q[7];
rz(1.6402714786797237) q[7];
ry(-2.944884681569641) q[8];
rz(-2.376902761237189) q[8];
ry(1.8864110169966781) q[9];
rz(1.8563189329761052) q[9];
ry(-1.2716015267445666) q[10];
rz(0.18019380337965796) q[10];
ry(1.3570021342144358) q[11];
rz(-0.3860798123636746) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.570833409118776) q[0];
rz(-0.16130518211739134) q[0];
ry(-3.14042695396815) q[1];
rz(-1.6884753890510806) q[1];
ry(-3.130034198516034) q[2];
rz(-3.0538611330436893) q[2];
ry(1.2319704214104714) q[3];
rz(-1.4258329931571527) q[3];
ry(0.021650345645838825) q[4];
rz(2.621268895010672) q[4];
ry(-1.1100639998517376) q[5];
rz(0.017066983125056723) q[5];
ry(-1.6446226181429422) q[6];
rz(-1.723586526724608) q[6];
ry(-3.141282583882326) q[7];
rz(-1.2466873954819846) q[7];
ry(2.2909407020263397) q[8];
rz(0.8262620955582777) q[8];
ry(-0.3691213350393505) q[9];
rz(3.0853574533226165) q[9];
ry(0.016642641667207157) q[10];
rz(-1.4320998436656707) q[10];
ry(-1.405609922207895) q[11];
rz(-0.6140448554624385) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.9611305885373715) q[0];
rz(-3.056266600613082) q[0];
ry(3.1409896667754165) q[1];
rz(0.8909556140723636) q[1];
ry(1.5712817565627653) q[2];
rz(2.548474238787706) q[2];
ry(1.638456911516174) q[3];
rz(0.0017397853028325526) q[3];
ry(-3.140836047218177) q[4];
rz(-1.3062608128387185) q[4];
ry(1.5703748462102345) q[5];
rz(0.30431273496153477) q[5];
ry(0.026149228240164746) q[6];
rz(1.39365316589692) q[6];
ry(-0.003137703164523664) q[7];
rz(1.2401653911071762) q[7];
ry(1.8924654127673737) q[8];
rz(0.0037382301307600234) q[8];
ry(-2.031618327253888) q[9];
rz(3.119849113452964) q[9];
ry(-1.6707521006431012) q[10];
rz(1.7574508811947058) q[10];
ry(-2.1716579756331744) q[11];
rz(-0.08665798494934261) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.720533573584763) q[0];
rz(1.751367674895577) q[0];
ry(0.0008853019886807554) q[1];
rz(0.0005706290904474181) q[1];
ry(-3.1392970771859536) q[2];
rz(-2.1425798467180415) q[2];
ry(1.6341705660092698) q[3];
rz(0.29575933272335725) q[3];
ry(3.1333363531984437) q[4];
rz(-2.5990062215384406) q[4];
ry(1.532068030996538) q[5];
rz(3.0571292648240433) q[5];
ry(0.10809395552255305) q[6];
rz(0.23121829658231616) q[6];
ry(-1.5761544017616602) q[7];
rz(1.6146171486947978) q[7];
ry(-1.0172455394206272) q[8];
rz(-1.4169574845630386) q[8];
ry(1.629553849840812) q[9];
rz(-0.07674070936677423) q[9];
ry(-2.9061159936957295) q[10];
rz(-2.704639381131244) q[10];
ry(1.870505008554101) q[11];
rz(-1.6960356621981472) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.37330560185217526) q[0];
rz(0.877588564076377) q[0];
ry(-1.5637256335475593) q[1];
rz(-2.5076774498163723) q[1];
ry(-1.5844252220213026) q[2];
rz(-0.22330612184165807) q[2];
ry(-2.0109126080675788) q[3];
rz(1.4875224033898273) q[3];
ry(-3.140186366968251) q[4];
rz(2.666496878096261) q[4];
ry(-0.0035765362153696856) q[5];
rz(0.08278303778336403) q[5];
ry(0.0004148035977253884) q[6];
rz(0.2299080804370428) q[6];
ry(-0.0012368953322915543) q[7];
rz(1.5288423396928514) q[7];
ry(-1.5481094627646799) q[8];
rz(0.29324097958948214) q[8];
ry(1.575479051285109) q[9];
rz(-0.12581681695706148) q[9];
ry(0.7076615052317559) q[10];
rz(0.22392120691879963) q[10];
ry(1.5577707012550344) q[11];
rz(0.6306192638160996) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.2481071233680074) q[0];
rz(-0.04710666254526155) q[0];
ry(1.5700913784988406) q[1];
rz(0.2764752283508563) q[1];
ry(-3.1371306435634825) q[2];
rz(0.515456300076031) q[2];
ry(1.561612116120096) q[3];
rz(-1.5674443533901845) q[3];
ry(-0.8623498081663747) q[4];
rz(0.27871465790404315) q[4];
ry(-1.652275207830555) q[5];
rz(1.0858769227170253) q[5];
ry(-0.735046827718076) q[6];
rz(-2.5700135625414533) q[6];
ry(2.7593040107134237) q[7];
rz(-1.8723010355046046) q[7];
ry(-3.141496320584135) q[8];
rz(-1.4410348269789366) q[8];
ry(-1.3501243818920567) q[9];
rz(1.5993880068464723) q[9];
ry(1.563466675955148) q[10];
rz(-1.1548423267340893) q[10];
ry(-1.5716483242950936) q[11];
rz(2.616131622584405) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.0411310333942647) q[0];
rz(0.8656390453805137) q[0];
ry(-3.111374721838466) q[1];
rz(2.378605830776915) q[1];
ry(0.002775200078585045) q[2];
rz(0.06748210477701176) q[2];
ry(1.5694752524314275) q[3];
rz(0.9062898394552441) q[3];
ry(-0.0022171240820032523) q[4];
rz(0.47385013201099446) q[4];
ry(0.0009974824579952468) q[5];
rz(-1.5446717244962656) q[5];
ry(0.0001990828333710508) q[6];
rz(0.3266974867755694) q[6];
ry(-3.139724265218264) q[7];
rz(1.811540567581364) q[7];
ry(-0.04206172307021472) q[8];
rz(0.9502059493030456) q[8];
ry(-1.572988623478821) q[9];
rz(1.9198324487323264) q[9];
ry(-0.01734590690040339) q[10];
rz(-1.3031356720496088) q[10];
ry(2.9949271613146626) q[11];
rz(0.5275119208757696) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.0714073382462719) q[0];
rz(0.11611020652257525) q[0];
ry(-0.8107719232276489) q[1];
rz(1.716422226891675) q[1];
ry(-1.2844374561521246) q[2];
rz(-2.760303841626153) q[2];
ry(-2.2296065984268205) q[3];
rz(-1.6244063544619385) q[3];
ry(0.08171685879064228) q[4];
rz(-1.7294900784862164) q[4];
ry(-0.8985203479057883) q[5];
rz(-0.7122082162288299) q[5];
ry(2.516988115290883) q[6];
rz(-2.09168994047178) q[6];
ry(-2.5992459920514572) q[7];
rz(-2.0694748104442944) q[7];
ry(-0.7370237447332941) q[8];
rz(-0.6213506114297402) q[8];
ry(-1.0696821477949712) q[9];
rz(-2.5565094082978606) q[9];
ry(-2.6088894559451017) q[10];
rz(-1.8118594037208373) q[10];
ry(0.5274193859676268) q[11];
rz(-2.1027332830777086) q[11];