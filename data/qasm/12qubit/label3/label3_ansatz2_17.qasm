OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.30939673956811) q[0];
rz(2.182421238734867) q[0];
ry(-1.564484323398123) q[1];
rz(-2.0929621204053612) q[1];
ry(-3.1411198301628755) q[2];
rz(-1.987633511163625) q[2];
ry(3.1415133518442695) q[3];
rz(-1.827842100028386) q[3];
ry(3.1397986662111106) q[4];
rz(1.6410214891479311) q[4];
ry(0.0003505196044075731) q[5];
rz(2.2757821585111104) q[5];
ry(-3.1402885784479815) q[6];
rz(1.7968885220708426) q[6];
ry(2.626189344486809) q[7];
rz(-1.5600520303022103) q[7];
ry(-1.9999211979253317) q[8];
rz(2.6602925649519973) q[8];
ry(1.5626680806840216) q[9];
rz(3.0854457441382515) q[9];
ry(2.012495003858864) q[10];
rz(-2.9909226400286792) q[10];
ry(-2.5683677028054483) q[11];
rz(-2.4273694575518427) q[11];
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
ry(1.571634761047922) q[0];
rz(3.1366085626882008) q[0];
ry(-2.1800202078562694) q[1];
rz(0.7827212218816084) q[1];
ry(-3.116444561890579) q[2];
rz(2.839796944165044) q[2];
ry(-3.1340702701738348) q[3];
rz(-0.885886339394136) q[3];
ry(-2.2298976218408004) q[4];
rz(-3.009609251663219) q[4];
ry(-3.1403780902498393) q[5];
rz(-2.6930211890982854) q[5];
ry(-1.7334436972578708) q[6];
rz(0.7363324294087894) q[6];
ry(-0.0014817211032260857) q[7];
rz(-1.6546078117808094) q[7];
ry(-1.5583943178572517) q[8];
rz(0.00646571653380157) q[8];
ry(2.2219504813798405) q[9];
rz(1.5925897206385526) q[9];
ry(-1.5877500294778986) q[10];
rz(0.0012205334829920034) q[10];
ry(0.006260188160463365) q[11];
rz(0.40626817038192353) q[11];
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
ry(1.5663080249231591) q[0];
rz(1.4731869537013234) q[0];
ry(-1.5738023542469293) q[1];
rz(0.863583066732958) q[1];
ry(-0.0015829504523710016) q[2];
rz(-0.7994437053633046) q[2];
ry(-0.0007749059478653463) q[3];
rz(-1.8427783802950661) q[3];
ry(-0.006304022230000149) q[4];
rz(2.218730308763062) q[4];
ry(3.1413192142336457) q[5];
rz(-3.110414825937608) q[5];
ry(-0.002370422271512851) q[6];
rz(-1.2793635605079334) q[6];
ry(0.02449904815756913) q[7];
rz(1.9502020072082762) q[7];
ry(1.5724967275637889) q[8];
rz(3.063197101496617) q[8];
ry(-0.19111601273856665) q[9];
rz(3.081411851015532) q[9];
ry(-1.572463644813153) q[10];
rz(0.08160801293897768) q[10];
ry(0.32568129616611274) q[11];
rz(2.436109643310726) q[11];
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
ry(1.4679474021115577) q[0];
rz(0.8959288917542487) q[0];
ry(-2.6546901932860725) q[1];
rz(0.20346163801151507) q[1];
ry(1.5811244063756) q[2];
rz(-1.8254979865782124) q[2];
ry(1.5632955917282407) q[3];
rz(3.132393478796561) q[3];
ry(-0.7298284990636874) q[4];
rz(0.22724042902193717) q[4];
ry(1.570348239284252) q[5];
rz(3.1393447473618035) q[5];
ry(1.1467323305698023) q[6];
rz(0.7497591256989687) q[6];
ry(1.5760846106871396) q[7];
rz(-3.13292841268077) q[7];
ry(0.6175667645591192) q[8];
rz(-1.1856660541869932) q[8];
ry(-1.2588025166834684) q[9];
rz(-1.5602105277790825) q[9];
ry(1.5800021256041679) q[10];
rz(2.753444909057731) q[10];
ry(1.5739833322494499) q[11];
rz(1.9709277466411637) q[11];
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
ry(1.307072650708661) q[0];
rz(-2.5567174919330133) q[0];
ry(-1.3645602042880256) q[1];
rz(0.40053752059811837) q[1];
ry(0.0067633444991224475) q[2];
rz(0.22838987564879074) q[2];
ry(-2.162231948684613) q[3];
rz(-1.54172408845017) q[3];
ry(-1.5208585900532718) q[4];
rz(-2.637307301770423) q[4];
ry(-1.2744795480448543) q[5];
rz(3.053732356682643) q[5];
ry(-2.2968808733347377) q[6];
rz(-2.0094007538950764) q[6];
ry(-0.4099657473215491) q[7];
rz(-1.5976468967261281) q[7];
ry(2.69965725342879) q[8];
rz(-1.6326844691328466) q[8];
ry(0.40953270950617987) q[9];
rz(-1.7324783153951) q[9];
ry(-1.7615820476257644) q[10];
rz(-0.05712305344243305) q[10];
ry(3.131470461498735) q[11];
rz(0.40961361602214186) q[11];
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
ry(-1.293766421268316) q[0];
rz(2.426982307147325) q[0];
ry(1.2065201784132116) q[1];
rz(-1.7068852367227656) q[1];
ry(0.7314709079456744) q[2];
rz(-2.426656367858897) q[2];
ry(-2.9318944269757563) q[3];
rz(1.5947737475410229) q[3];
ry(-1.543951665193044) q[4];
rz(-0.6121748586448058) q[4];
ry(-0.00269607333927915) q[5];
rz(0.09058871212002967) q[5];
ry(-1.4103786011695583) q[6];
rz(-0.058309119544614596) q[6];
ry(-3.1121218292457273) q[7];
rz(1.5395751448270742) q[7];
ry(-2.2700930141159548) q[8];
rz(2.7384950964208894) q[8];
ry(-0.03326714127039508) q[9];
rz(-1.4223936939709452) q[9];
ry(-1.2240554981281448) q[10];
rz(1.864220068891917) q[10];
ry(1.4044964007744598) q[11];
rz(-1.5648141137727738) q[11];
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
ry(-0.5544006668187046) q[0];
rz(0.4148089277549061) q[0];
ry(-0.8214669741796419) q[1];
rz(1.5979978997979565) q[1];
ry(-3.1202599452613) q[2];
rz(-2.477274109054428) q[2];
ry(-2.80443120395016) q[3];
rz(1.5547950094771588) q[3];
ry(1.134613543622484) q[4];
rz(2.4818906941336896) q[4];
ry(-2.610614823136754) q[5];
rz(1.5728621369664033) q[5];
ry(-2.6691184302954514) q[6];
rz(0.11016703918942745) q[6];
ry(2.6409076831768172) q[7];
rz(-1.5821720816916331) q[7];
ry(-2.8349990413680497) q[8];
rz(0.5452002568123858) q[8];
ry(-0.49533818296242865) q[9];
rz(1.5845853832198216) q[9];
ry(3.010392871786005) q[10];
rz(2.0759250981268575) q[10];
ry(-1.708074719278093) q[11];
rz(-0.21313982791456795) q[11];
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
ry(-2.567716730788998) q[0];
rz(0.4785228886210975) q[0];
ry(2.4498244673176486) q[1];
rz(0.07651287261968333) q[1];
ry(2.614072645687922) q[2];
rz(1.8274472268621098) q[2];
ry(-2.298956235944007) q[3];
rz(-1.5570579244581164) q[3];
ry(0.4723950545215745) q[4];
rz(2.9542108129946536) q[4];
ry(1.9734917264860687) q[5];
rz(-1.5689093698442105) q[5];
ry(-1.9190736625223987) q[6];
rz(-2.3102085182592265) q[6];
ry(2.361624867112122) q[7];
rz(1.5658064368351663) q[7];
ry(1.6919505963542365) q[8];
rz(-2.7324393150224195) q[8];
ry(0.7860530132422738) q[9];
rz(-1.5813207145695949) q[9];
ry(-2.964295511635462) q[10];
rz(-1.3479620572926514) q[10];
ry(-0.011883847231461075) q[11];
rz(1.4263775760897541) q[11];
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
ry(-0.360707852437221) q[0];
rz(1.920345057637726) q[0];
ry(1.9198098553194396) q[1];
rz(-2.7395344319926473) q[1];
ry(0.012136904889503761) q[2];
rz(1.8196471133200953) q[2];
ry(0.16504890139569728) q[3];
rz(-1.5873481328909709) q[3];
ry(2.4183967732599276) q[4];
rz(0.48367828584042727) q[4];
ry(-2.091865672479069) q[5];
rz(0.3124887754603988) q[5];
ry(1.8690811896554278) q[6];
rz(-2.136133138621174) q[6];
ry(0.2643101219800301) q[7];
rz(1.5671131133503335) q[7];
ry(0.7137396481285055) q[8];
rz(0.019103320638809155) q[8];
ry(0.2600215474737581) q[9];
rz(1.5794091022902992) q[9];
ry(-1.1235647282615333) q[10];
rz(1.069704760756279) q[10];
ry(-0.012077894370545295) q[11];
rz(0.37468419234772554) q[11];
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
ry(0.9849010887186891) q[0];
rz(-0.306355199549608) q[0];
ry(-2.4084261178225734) q[1];
rz(0.1399018552007106) q[1];
ry(-3.1138004078812824) q[2];
rz(2.585185590503467) q[2];
ry(0.8380626240644065) q[3];
rz(-1.5823403551652593) q[3];
ry(-0.17452916943897817) q[4];
rz(-0.06128983658607101) q[4];
ry(0.0010247504877227249) q[5];
rz(-0.3230915816634357) q[5];
ry(1.997610350874754) q[6];
rz(-2.361507808264463) q[6];
ry(0.23154714037229765) q[7];
rz(2.981747683447103) q[7];
ry(0.45440240514265806) q[8];
rz(-0.5642184713981244) q[8];
ry(0.23293997852563028) q[9];
rz(1.9946451414661714) q[9];
ry(1.0168037905328833) q[10];
rz(2.3472440871174545) q[10];
ry(2.678486906008847) q[11];
rz(-1.5138464666790359) q[11];
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
ry(1.2351584783623313) q[0];
rz(-0.31598582114358464) q[0];
ry(1.1451670135204193) q[1];
rz(-0.3955756461039348) q[1];
ry(-0.0011630232273313783) q[2];
rz(0.5590801447357387) q[2];
ry(2.7052512632927246) q[3];
rz(-2.9962856319910256) q[3];
ry(0.45281971890952993) q[4];
rz(2.6268126286017837) q[4];
ry(2.8137519997257994) q[5];
rz(-1.5779867395004432) q[5];
ry(-1.2503411797325201) q[6];
rz(-2.69151449726155) q[6];
ry(-3.1409078690261976) q[7];
rz(-0.20241603615228068) q[7];
ry(-2.7488137652266635) q[8];
rz(-0.9096573016393651) q[8];
ry(3.139624764262798) q[9];
rz(-1.1574981406358034) q[9];
ry(-2.2145882768407077) q[10];
rz(2.3046984430176805) q[10];
ry(-2.817305233790647) q[11];
rz(-1.35814492567793) q[11];
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
ry(-1.8912255033048284) q[0];
rz(-0.9454972865137643) q[0];
ry(-1.2945947050606135) q[1];
rz(1.6577589838908924) q[1];
ry(0.052568853444132514) q[2];
rz(-0.9877054856948311) q[2];
ry(3.136146956152091) q[3];
rz(0.15477025199675865) q[3];
ry(-2.8140495689927905) q[4];
rz(-2.910532157985122) q[4];
ry(-2.70951065068124) q[5];
rz(1.5725742740799207) q[5];
ry(2.896600436059296) q[6];
rz(0.8019990954658226) q[6];
ry(0.22112141750624126) q[7];
rz(-1.2321686601658453) q[7];
ry(-0.7718465606960097) q[8];
rz(2.308791221605562) q[8];
ry(0.22011143661629262) q[9];
rz(-1.3878985284818697) q[9];
ry(-2.1444502630552402) q[10];
rz(-1.5762017140146494) q[10];
ry(-3.115876366778902) q[11];
rz(1.752590236708395) q[11];
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
ry(-1.3060567986728566) q[0];
rz(-1.5151032778738072) q[0];
ry(0.9773256617258415) q[1];
rz(-2.795703399453291) q[1];
ry(-0.6150502227868672) q[2];
rz(-1.5996219625765473) q[2];
ry(1.0346382049514755) q[3];
rz(-0.014926174334255471) q[3];
ry(-1.5967341139124) q[4];
rz(-2.8124675891606365) q[4];
ry(2.677549973050967) q[5];
rz(1.571412185684549) q[5];
ry(-2.4145202428073427) q[6];
rz(0.449829255025639) q[6];
ry(-3.122617733440394) q[7];
rz(1.8630341963033237) q[7];
ry(1.6579045567461363) q[8];
rz(-1.890441621868473) q[8];
ry(-3.124044658272267) q[9];
rz(-1.407402736504314) q[9];
ry(0.14539544198441823) q[10];
rz(-0.06632465220560557) q[10];
ry(2.4554937258492266) q[11];
rz(-0.33396121121409705) q[11];
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
ry(-1.8146655108523326) q[0];
rz(-1.4564920462315414) q[0];
ry(2.2645765823504282) q[1];
rz(-2.8706265114814973) q[1];
ry(-0.7001059635052764) q[2];
rz(1.555969891844013) q[2];
ry(-1.5985783816160088) q[3];
rz(2.6318298865558223) q[3];
ry(-0.728855267567999) q[4];
rz(2.2467369364248584) q[4];
ry(-2.5606357860750846) q[5];
rz(-1.5698884886945694) q[5];
ry(-1.6542630425981022) q[6];
rz(-1.926386024736503) q[6];
ry(1.2307753563129349) q[7];
rz(-0.703535289703339) q[7];
ry(-2.1533560412436845) q[8];
rz(0.6673367962166372) q[8];
ry(-1.2374232713817188) q[9];
rz(-2.693537394061157) q[9];
ry(-1.1712092564178782) q[10];
rz(-0.9263382547658084) q[10];
ry(3.1365599991395894) q[11];
rz(1.3781685401171693) q[11];
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
ry(-0.005798422072641252) q[0];
rz(-1.6899048636598977) q[0];
ry(-1.565183295005781) q[1];
rz(2.895772869463323) q[1];
ry(3.0804362569987083) q[2];
rz(1.555024232033328) q[2];
ry(-0.9012503210702079) q[3];
rz(-1.31061358850138) q[3];
ry(-3.1362295495776116) q[4];
rz(-2.5316508177497647) q[4];
ry(1.6408998264190933) q[5];
rz(-1.5754277599933353) q[5];
ry(0.0007867369220833704) q[6];
rz(-0.5081379747263535) q[6];
ry(3.1319906814174585) q[7];
rz(-0.7001752783315993) q[7];
ry(3.138807239856117) q[8];
rz(-3.0869622692367336) q[8];
ry(-3.3449097234584294e-05) q[9];
rz(2.6930931579249076) q[9];
ry(-0.00037114450312913645) q[10];
rz(-3.0874577323717345) q[10];
ry(3.139091797872976) q[11];
rz(-2.9881200520121145) q[11];
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
ry(1.3667507809499297) q[0];
rz(0.3797378284951858) q[0];
ry(0.006055110356655291) q[1];
rz(0.25498568737008415) q[1];
ry(0.6447772444412684) q[2];
rz(1.9674589064182004) q[2];
ry(1.5652934809913812) q[3];
rz(0.21683887692032489) q[3];
ry(1.582583510081552) q[4];
rz(-0.8019903754250279) q[4];
ry(0.6364308709743263) q[5];
rz(1.5724825879336095) q[5];
ry(-1.7306421614792689) q[6];
rz(-0.9528329077498833) q[6];
ry(-0.463232797559526) q[7];
rz(1.5630460714954084) q[7];
ry(1.2473099754146775) q[8];
rz(1.2888579165696845) q[8];
ry(-0.4639390177342868) q[9];
rz(-1.5667155427805284) q[9];
ry(0.9537533365193811) q[10];
rz(-0.36581411519842116) q[10];
ry(-1.650931451555662) q[11];
rz(1.6573832689533465) q[11];
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
ry(-1.5603829993862082) q[0];
rz(-1.3574069330488796) q[0];
ry(-2.1546394166626586) q[1];
rz(3.1307595692248897) q[1];
ry(3.1318674215042575) q[2];
rz(0.3598249830773819) q[2];
ry(0.005241503822261961) q[3];
rz(2.897662510018414) q[3];
ry(-0.6561857698758929) q[4];
rz(2.030458806552991) q[4];
ry(2.2742154843140634) q[5];
rz(0.2189980777137351) q[5];
ry(1.2118784802568667) q[6];
rz(-0.847202803435283) q[6];
ry(0.9929649457215182) q[7];
rz(1.554828979568506) q[7];
ry(1.0060907327611983) q[8];
rz(-0.5306706994351504) q[8];
ry(-0.9987166526661774) q[9];
rz(1.5662195316764136) q[9];
ry(2.4256520710235456) q[10];
rz(0.1772608108283098) q[10];
ry(0.026345347851281592) q[11];
rz(-1.659838309309281) q[11];
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
ry(0.9644115006306316) q[0];
rz(2.4208441865321033) q[0];
ry(1.6067600060913294) q[1];
rz(2.294755043511559) q[1];
ry(-1.6812587790728832) q[2];
rz(1.6387927400153144) q[2];
ry(-1.571218519396325) q[3];
rz(0.7745429182148922) q[3];
ry(-0.8770611314392412) q[4];
rz(-2.5900928823839644) q[4];
ry(3.135553322176045) q[5];
rz(0.22462711656835044) q[5];
ry(-2.8818844485942128) q[6];
rz(-0.9322920512763393) q[6];
ry(2.3577940337381262) q[7];
rz(-1.8249372976144596) q[7];
ry(-1.1609849859988033) q[8];
rz(0.02946258904413312) q[8];
ry(-2.3568735288954574) q[9];
rz(-1.2302531749606462) q[9];
ry(2.025798078931997) q[10];
rz(-0.8644446394934286) q[10];
ry(-1.0615170059591454) q[11];
rz(0.9814752160216502) q[11];
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
ry(-1.7526809415957123) q[0];
rz(1.533896811444853) q[0];
ry(0.009216690456513991) q[1];
rz(1.6199484294917452) q[1];
ry(-1.2291132106111924) q[2];
rz(2.3959881186893894) q[2];
ry(-0.0048683330202867895) q[3];
rz(-0.23412811571166792) q[3];
ry(-0.07128517454202958) q[4];
rz(-1.431269102763488) q[4];
ry(1.5518185140141407) q[5];
rz(-1.5699896736986467) q[5];
ry(2.8996552497131085) q[6];
rz(-1.3989959685630253) q[6];
ry(-0.004374976990075504) q[7];
rz(-1.3356721562578435) q[7];
ry(-3.001269238695413) q[8];
rz(-2.0155895868730047) q[8];
ry(-3.1399466266858402) q[9];
rz(-1.1974310685369112) q[9];
ry(-3.011761113779002) q[10];
rz(1.4747485837339562) q[10];
ry(3.1390684152549397) q[11];
rz(-2.140019628542528) q[11];
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
ry(-2.0180621907131315) q[0];
rz(1.8124130095355646) q[0];
ry(-0.0013490285047668138) q[1];
rz(-2.5374366388394676) q[1];
ry(1.5673313517516374) q[2];
rz(2.540372117629486) q[2];
ry(-3.138908524454633) q[3];
rz(2.9610056679846277) q[3];
ry(1.4832664502007051) q[4];
rz(-2.5255001415643807) q[4];
ry(0.529498786119778) q[5];
rz(0.6171906671199651) q[5];
ry(1.1603776585139354) q[6];
rz(-3.000758487829261) q[6];
ry(-0.07609871816686642) q[7];
rz(0.35413662901082066) q[7];
ry(-0.5179124910063034) q[8];
rz(-1.76087516626504) q[8];
ry(-3.0677902704632847) q[9];
rz(2.83770068139015) q[9];
ry(-2.552295667254597) q[10];
rz(1.887105119301939) q[10];
ry(-2.944317381987103) q[11];
rz(-2.213888326911597) q[11];
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
ry(1.5368137254544105) q[0];
rz(-2.511761456032836) q[0];
ry(-1.1187804206291228) q[1];
rz(2.000776381029927) q[1];
ry(-0.5359492466297979) q[2];
rz(-1.9457310022163057) q[2];
ry(-2.0206269467603075) q[3];
rz(-1.1336955716984078) q[3];
ry(-1.3292481225472994) q[4];
rz(0.24296591819217048) q[4];
ry(-2.458531381885305) q[5];
rz(2.6502855783895996) q[5];
ry(2.959215183683323) q[6];
rz(-0.6978700774575337) q[6];
ry(-0.5788467438321104) q[7];
rz(-1.642465028192529) q[7];
ry(2.756184532609235) q[8];
rz(-0.20494482356504518) q[8];
ry(-2.5525648437088893) q[9];
rz(1.4949578736265294) q[9];
ry(-1.1689320651524389) q[10];
rz(0.19057410138978256) q[10];
ry(2.089355012498726) q[11];
rz(-0.06865722735906399) q[11];