OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.3359446742135517) q[0];
rz(0.3468266319877232) q[0];
ry(2.430500049169977) q[1];
rz(-1.2581593022254582) q[1];
ry(2.448133119634433) q[2];
rz(0.8501000032094866) q[2];
ry(2.1926522290702435) q[3];
rz(1.9432877421498587) q[3];
ry(-2.9286501914872787) q[4];
rz(-0.4948970949319334) q[4];
ry(0.1940830476203903) q[5];
rz(0.3714996922301678) q[5];
ry(3.1107730037777532) q[6];
rz(-2.1348975613093972) q[6];
ry(0.03148960370732912) q[7];
rz(-1.8014033464253654) q[7];
ry(2.031425609655018) q[8];
rz(-2.090080712297654) q[8];
ry(-0.9249833829759089) q[9];
rz(-2.1097541855608544) q[9];
ry(1.549056842384365) q[10];
rz(3.022289554509706) q[10];
ry(0.5631168754918479) q[11];
rz(1.261671079984895) q[11];
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
ry(0.7622976377878247) q[0];
rz(1.360332304529786) q[0];
ry(-1.7576522865943263) q[1];
rz(1.4172939880683928) q[1];
ry(-1.8990013313178935) q[2];
rz(1.5885100189817292) q[2];
ry(-1.3269440343210475) q[3];
rz(-2.506524228722263) q[3];
ry(-1.406415764321852) q[4];
rz(-1.3873340692019767) q[4];
ry(1.681019554630411) q[5];
rz(2.7473471899676642) q[5];
ry(1.866713943968028) q[6];
rz(1.5516241687001129) q[6];
ry(1.6004387208040152) q[7];
rz(0.650314034788144) q[7];
ry(2.0282826836308) q[8];
rz(2.153991882557669) q[8];
ry(2.9092908729323663) q[9];
rz(1.8402145415059215) q[9];
ry(-1.8627802823300896) q[10];
rz(0.04205514432492998) q[10];
ry(1.4853779671566525) q[11];
rz(-1.1083789695032262) q[11];
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
ry(-1.751581352451729) q[0];
rz(-1.7581826125344586) q[0];
ry(1.070572634894917) q[1];
rz(2.047241707626463) q[1];
ry(1.6644216281192588) q[2];
rz(-0.7055841130369156) q[2];
ry(-2.276823361365074) q[3];
rz(3.040083949777408) q[3];
ry(-3.0938160570789375) q[4];
rz(-0.8933684792060719) q[4];
ry(-0.044189680868355966) q[5];
rz(1.7981500093478493) q[5];
ry(-0.0020795020071489854) q[6];
rz(-0.21835923041258634) q[6];
ry(3.137154390134198) q[7];
rz(2.1030330180952648) q[7];
ry(1.662219405912454) q[8];
rz(-1.019585033155668) q[8];
ry(0.5573196563736892) q[9];
rz(-1.5457255581916596) q[9];
ry(2.478062263977569) q[10];
rz(3.121906958305016) q[10];
ry(-0.4759271186370701) q[11];
rz(1.504815031304923) q[11];
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
ry(-2.3924206066229576) q[0];
rz(-3.139387188735457) q[0];
ry(-2.2306007968198536) q[1];
rz(-0.816780288165209) q[1];
ry(0.6046289641311864) q[2];
rz(1.7778620243736434) q[2];
ry(-1.6649281607938626) q[3];
rz(2.8306345731613582) q[3];
ry(0.21915272758090934) q[4];
rz(0.39743150189018905) q[4];
ry(-1.7074691899392478) q[5];
rz(-2.3058008505370178) q[5];
ry(-2.5046845832269518) q[6];
rz(-1.6816615693436) q[6];
ry(0.7116178199082359) q[7];
rz(-0.2944491281888526) q[7];
ry(1.6151039655621693) q[8];
rz(1.7605987344396206) q[8];
ry(-2.3015572901128603) q[9];
rz(1.0846196334419131) q[9];
ry(1.0819524978047699) q[10];
rz(2.454492441336645) q[10];
ry(-2.3843388227072695) q[11];
rz(-1.8868678529963463) q[11];
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
ry(0.35911215498867044) q[0];
rz(-0.07569155356213184) q[0];
ry(-0.6989237833145889) q[1];
rz(-2.7000377188226876) q[1];
ry(-1.519149191397465) q[2];
rz(-1.5685278768814592) q[2];
ry(-1.8681682687970715) q[3];
rz(-0.6199056403849913) q[3];
ry(3.1292238928549656) q[4];
rz(1.0712788068453396) q[4];
ry(0.034694891022338886) q[5];
rz(-2.3917913417600154) q[5];
ry(0.006352301402651861) q[6];
rz(-1.6959560882215943) q[6];
ry(-3.138706374613929) q[7];
rz(-0.483317743678988) q[7];
ry(1.2940318184484043) q[8];
rz(-2.5572759877381297) q[8];
ry(1.0924617770189693) q[9];
rz(-1.272377246846541) q[9];
ry(-1.812119996358922) q[10];
rz(-0.5954280091040829) q[10];
ry(-1.2459702940284152) q[11];
rz(3.1304301252576856) q[11];
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
ry(-1.0682060857138218) q[0];
rz(-1.2511472001567985) q[0];
ry(1.7420219078458619) q[1];
rz(-0.2873611520819832) q[1];
ry(-1.390952416436254) q[2];
rz(1.6908058829221975) q[2];
ry(2.3453200923401387) q[3];
rz(1.875732735019642) q[3];
ry(-1.1449283747855716) q[4];
rz(2.6257186028891932) q[4];
ry(-1.2231077026095192) q[5];
rz(-1.8780557741812398) q[5];
ry(1.9733629332361808) q[6];
rz(2.224278742302589) q[6];
ry(1.9211372612309132) q[7];
rz(-2.33279557507395) q[7];
ry(0.06533139814756839) q[8];
rz(-0.11559386387806468) q[8];
ry(-2.398306224381732) q[9];
rz(0.6188471528772714) q[9];
ry(2.0976445714298975) q[10];
rz(-2.0474704594932325) q[10];
ry(3.024907384339642) q[11];
rz(-1.6177025601803496) q[11];
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
ry(0.25943922863615665) q[0];
rz(0.641154699416621) q[0];
ry(2.539237997552682) q[1];
rz(0.5788211244504309) q[1];
ry(3.0734905965912476) q[2];
rz(-0.4511881784650651) q[2];
ry(2.6749052535266977) q[3];
rz(-2.9858709165461756) q[3];
ry(1.775249172465058) q[4];
rz(1.2085692483577404) q[4];
ry(2.6936628486531733) q[5];
rz(-0.022223273280537192) q[5];
ry(-3.140279810728899) q[6];
rz(-0.08583055040671668) q[6];
ry(3.1391337369699617) q[7];
rz(-1.8465538828481707) q[7];
ry(2.1891181761008798) q[8];
rz(0.5776686174649216) q[8];
ry(-1.8236142141231275) q[9];
rz(-0.060410124619832424) q[9];
ry(-1.5565123249987574) q[10];
rz(-1.1812681374104113) q[10];
ry(-1.5099643898393167) q[11];
rz(2.499459250679197) q[11];
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
ry(0.6471987334844345) q[0];
rz(-2.541949869829649) q[0];
ry(2.1176355977467676) q[1];
rz(2.4857323485061324) q[1];
ry(-1.5957866555397038) q[2];
rz(-2.7507841865736604) q[2];
ry(-2.0794381225286016) q[3];
rz(0.08367774360455704) q[3];
ry(-1.7558773001335342) q[4];
rz(-2.138402753383996) q[4];
ry(-1.3012847622771615) q[5];
rz(1.7954106656397064) q[5];
ry(-3.1402025663525293) q[6];
rz(2.875576845670831) q[6];
ry(-3.1380533168756997) q[7];
rz(0.30041377873026615) q[7];
ry(2.280385339225186) q[8];
rz(-1.9417807995001342) q[8];
ry(-1.732154497747511) q[9];
rz(-2.966689358714187) q[9];
ry(-2.499418586498328) q[10];
rz(-2.4597044602650278) q[10];
ry(-0.4935601847554283) q[11];
rz(-1.425326462056789) q[11];
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
ry(0.4991902251385189) q[0];
rz(-2.67973128927151) q[0];
ry(1.0256279987466648) q[1];
rz(-0.8134683298296315) q[1];
ry(-2.328038729956009) q[2];
rz(-3.1244884016002232) q[2];
ry(1.7344100005279959) q[3];
rz(2.723911994443034) q[3];
ry(0.10098177286853478) q[4];
rz(-2.080690217247258) q[4];
ry(2.3565371136192486) q[5];
rz(1.6725442427505708) q[5];
ry(-0.00683887676399042) q[6];
rz(3.0331367704317245) q[6];
ry(3.1365252914006745) q[7];
rz(1.4318181218933566) q[7];
ry(-1.0038014773404411) q[8];
rz(-0.5070838928356725) q[8];
ry(2.6027829258905224) q[9];
rz(0.1918918947091545) q[9];
ry(-1.6126187796570113) q[10];
rz(-2.1680723672629165) q[10];
ry(-0.7239757145124353) q[11];
rz(2.178574413376131) q[11];
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
ry(-2.4979975117733573) q[0];
rz(2.8445091870230175) q[0];
ry(-0.6591142592092325) q[1];
rz(1.0124496305423003) q[1];
ry(0.5247830368906) q[2];
rz(-0.5112081864251972) q[2];
ry(-2.5022796530190257) q[3];
rz(-0.9411651330087291) q[3];
ry(2.521653070306157) q[4];
rz(-2.0221404253212403) q[4];
ry(0.08932994670498129) q[5];
rz(-0.7120836469898482) q[5];
ry(1.5673307659452966) q[6];
rz(-2.184744878606138) q[6];
ry(-1.5671676098309903) q[7];
rz(2.9429292065075665) q[7];
ry(-1.6867835919586085) q[8];
rz(1.8581194549948878) q[8];
ry(1.9526265198904156) q[9];
rz(2.4716973946415637) q[9];
ry(1.5975820676271382) q[10];
rz(3.0045524870393283) q[10];
ry(-1.4629978603712253) q[11];
rz(2.6952205367341384) q[11];
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
ry(-1.923925894467412) q[0];
rz(-2.3038468522524296) q[0];
ry(-1.3746984975808352) q[1];
rz(2.138376186174407) q[1];
ry(0.8168276457409203) q[2];
rz(-1.2331059029256877) q[2];
ry(1.6595676015139889) q[3];
rz(1.6442897128552332) q[3];
ry(-0.01865845484592328) q[4];
rz(2.093755426447106) q[4];
ry(3.0988168565614207) q[5];
rz(-2.671038107170854) q[5];
ry(-3.138695600543627) q[6];
rz(0.4090219866547676) q[6];
ry(0.00042754649714776236) q[7];
rz(1.1180900636763804) q[7];
ry(1.5286069072884467) q[8];
rz(0.1905083091310651) q[8];
ry(2.818557741704226) q[9];
rz(-2.468486704579583) q[9];
ry(1.0324890839481107) q[10];
rz(-1.4603530281642367) q[10];
ry(-0.4966580629911881) q[11];
rz(2.566501015053557) q[11];
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
ry(1.7080137933069397) q[0];
rz(0.6612652265448186) q[0];
ry(-2.0810624721741067) q[1];
rz(1.3795716691572846) q[1];
ry(-1.3952488924768272) q[2];
rz(-2.6247630227029086) q[2];
ry(0.5417961566812552) q[3];
rz(2.4886622310876945) q[3];
ry(-2.77954615002097) q[4];
rz(-1.7721820552003562) q[4];
ry(-2.9125465440603713) q[5];
rz(-1.7936731862326214) q[5];
ry(1.5978578417384954) q[6];
rz(-1.7164398433094057) q[6];
ry(-1.5569130862694247) q[7];
rz(0.07867339720538347) q[7];
ry(0.9379036720353223) q[8];
rz(1.3970190195814307) q[8];
ry(-2.8956019462753426) q[9];
rz(-0.8876936979164208) q[9];
ry(0.656115860305939) q[10];
rz(-1.5318249620971018) q[10];
ry(1.5888643574716266) q[11];
rz(-2.3918545032251486) q[11];
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
ry(1.4251650171109864) q[0];
rz(1.5880277256919333) q[0];
ry(-1.31730210757694) q[1];
rz(0.9087392651176184) q[1];
ry(0.5879084089779196) q[2];
rz(-2.3959732122548445) q[2];
ry(-1.2386623940786785) q[3];
rz(-0.022039244773902244) q[3];
ry(0.008583527478670201) q[4];
rz(-2.2108680777624485) q[4];
ry(-0.025599736546194407) q[5];
rz(1.1378043525448378) q[5];
ry(3.1363837294754506) q[6];
rz(2.8405116139640825) q[6];
ry(0.003858422030431185) q[7];
rz(-1.6677786633930838) q[7];
ry(-1.9905552662261465) q[8];
rz(-0.19262484302461846) q[8];
ry(-1.2238870995520368) q[9];
rz(3.117197813178978) q[9];
ry(2.6214654593845967) q[10];
rz(-2.6840126529536295) q[10];
ry(2.988010647368042) q[11];
rz(-2.740247146578825) q[11];
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
ry(-2.4104007896847204) q[0];
rz(1.985615749031718) q[0];
ry(1.7912033054449021) q[1];
rz(2.9232177893526288) q[1];
ry(0.8958996986743736) q[2];
rz(-2.0493927811773887) q[2];
ry(-1.7736675460025222) q[3];
rz(-3.034643825699735) q[3];
ry(0.050828255837582645) q[4];
rz(1.32128227470593) q[4];
ry(0.0435180553161425) q[5];
rz(-0.3039494834910687) q[5];
ry(-3.028305648071936) q[6];
rz(-0.28578058348563123) q[6];
ry(-2.961460406406453) q[7];
rz(-0.05832944992422229) q[7];
ry(-2.0411242559467664) q[8];
rz(-0.23649643920871846) q[8];
ry(1.748507757266867) q[9];
rz(-0.24219368455765175) q[9];
ry(0.11421135011118212) q[10];
rz(-0.8645697563162128) q[10];
ry(0.046098479488304006) q[11];
rz(-1.851834486153356) q[11];
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
ry(1.9657558553106211) q[0];
rz(2.8248482809636926) q[0];
ry(-1.9196045262077268) q[1];
rz(2.9501876368390603) q[1];
ry(0.8825861539941657) q[2];
rz(-0.6700660802858049) q[2];
ry(-1.6446125389322237) q[3];
rz(0.7717398658142187) q[3];
ry(3.131276964049944) q[4];
rz(2.0603455835782434) q[4];
ry(3.1289348239882413) q[5];
rz(-2.9732739840956355) q[5];
ry(-0.05078954093724276) q[6];
rz(-2.798310182156141) q[6];
ry(-1.568129009362635) q[7];
rz(-1.8171436157134886) q[7];
ry(0.3114666973837196) q[8];
rz(-1.0892635128649104) q[8];
ry(0.6223681912714055) q[9];
rz(0.06634150178151275) q[9];
ry(1.0792195022207358) q[10];
rz(0.6702007939100056) q[10];
ry(-2.2781074958020984) q[11];
rz(-1.7801078657766949) q[11];
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
ry(-1.7323237252800048) q[0];
rz(-2.801333488798104) q[0];
ry(-2.1909205614263882) q[1];
rz(-0.8730271542302146) q[1];
ry(1.0475730163676742) q[2];
rz(-1.630381363520739) q[2];
ry(2.593996239226402) q[3];
rz(1.7267044798898001) q[3];
ry(-1.5951752468841418) q[4];
rz(0.3887848210787356) q[4];
ry(3.1322527764581167) q[5];
rz(-2.9908204923160397) q[5];
ry(-1.5746784411517822) q[6];
rz(-1.977361003889751) q[6];
ry(1.9878883390018816) q[7];
rz(-0.7319413288127423) q[7];
ry(3.119980990175112) q[8];
rz(-2.463815809297509) q[8];
ry(0.0015329410829636458) q[9];
rz(2.453259694594898) q[9];
ry(1.1783840369822256) q[10];
rz(2.023639221095496) q[10];
ry(2.665703778962234) q[11];
rz(0.7188224887748618) q[11];
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
ry(-0.30242595272103046) q[0];
rz(-0.9506161436884731) q[0];
ry(0.6171894726529997) q[1];
rz(3.0171728781688554) q[1];
ry(3.1379869208240567) q[2];
rz(-2.250967296568528) q[2];
ry(-1.5746174460563211) q[3];
rz(-1.608398008726261) q[3];
ry(0.0006860649701021515) q[4];
rz(-0.4051461732811603) q[4];
ry(0.0020051099011955094) q[5];
rz(2.377764755077659) q[5];
ry(3.101294907301725) q[6];
rz(2.7645081278053483) q[6];
ry(0.006800027057767681) q[7];
rz(-0.765523981067426) q[7];
ry(3.112478306938928) q[8];
rz(-0.4880178110423868) q[8];
ry(3.134147290425322) q[9];
rz(-1.571229876927383) q[9];
ry(1.1285970819312796) q[10];
rz(3.038136666515125) q[10];
ry(0.7674360264670684) q[11];
rz(-1.2821185005976918) q[11];
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
ry(-0.012732885598143005) q[0];
rz(1.3649820103738852) q[0];
ry(-1.5973074284407291) q[1];
rz(-1.5837345623235555) q[1];
ry(-3.1373724926926125) q[2];
rz(2.4986049569003708) q[2];
ry(1.558077752004036) q[3];
rz(0.6030774004791972) q[3];
ry(-1.5037555574940986) q[4];
rz(1.4871938343782003) q[4];
ry(-1.5681630488526181) q[5];
rz(-1.6376767961879068) q[5];
ry(0.1756057171597198) q[6];
rz(-0.00569240246137781) q[6];
ry(1.8856204526371645) q[7];
rz(-0.11794247288401571) q[7];
ry(0.010015901503898774) q[8];
rz(-1.1191629255927227) q[8];
ry(-3.1338813659473357) q[9];
rz(2.0310964034480916) q[9];
ry(0.5234587263377134) q[10];
rz(2.0662487904004085) q[10];
ry(-2.409419142655996) q[11];
rz(2.1127331307001005) q[11];
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
ry(2.4191045861988654) q[0];
rz(-2.366750208588228) q[0];
ry(-1.3667677034610888) q[1];
rz(3.136064363772689) q[1];
ry(-0.013309848586419858) q[2];
rz(-0.0713559999774765) q[2];
ry(1.5731670461901954) q[3];
rz(-1.5601250962574054) q[3];
ry(0.0006064989056049259) q[4];
rz(1.5682367473437129) q[4];
ry(-0.007779690653577198) q[5];
rz(-1.2760808509519777) q[5];
ry(-0.006412580550097273) q[6];
rz(-2.9459897786100253) q[6];
ry(-3.1365652284869614) q[7];
rz(0.27490461601392546) q[7];
ry(1.5748607282784723) q[8];
rz(0.2745835851095841) q[8];
ry(1.5740645667672606) q[9];
rz(1.6055911270146244) q[9];
ry(-0.6333621070139716) q[10];
rz(-1.114861992126905) q[10];
ry(-0.36921384406677227) q[11];
rz(2.711806285758871) q[11];
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
ry(0.0017196674713205694) q[0];
rz(2.746495194821205) q[0];
ry(-1.5576148167106953) q[1];
rz(2.8859828892403736) q[1];
ry(2.4612226249901465) q[2];
rz(-1.484692221168639) q[2];
ry(-2.29764581801918) q[3];
rz(3.133661430239524) q[3];
ry(-3.1284045910997333) q[4];
rz(0.8623837697163301) q[4];
ry(-3.113555322625041) q[5];
rz(-1.2973911033553467) q[5];
ry(0.03071749925740619) q[6];
rz(2.038680535858057) q[6];
ry(1.5406252132642475) q[7];
rz(2.0605969843253336) q[7];
ry(0.010846445664102285) q[8];
rz(-2.024689369299648) q[8];
ry(1.5585207538225148) q[9];
rz(1.575198924005062) q[9];
ry(-1.7615089896676395) q[10];
rz(-1.1765925756586277) q[10];
ry(0.07420371761303275) q[11];
rz(-0.26674955705038794) q[11];
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
ry(-3.1318806902067813) q[0];
rz(-1.7594994370922619) q[0];
ry(-1.8035439619220552) q[1];
rz(-3.1021475935467726) q[1];
ry(2.926119744258036) q[2];
rz(2.0727029924342757) q[2];
ry(-1.5769757560913131) q[3];
rz(1.3891202323132632) q[3];
ry(0.007210402896193457) q[4];
rz(0.6325688103252451) q[4];
ry(3.140054207646776) q[5];
rz(2.4471384200917887) q[5];
ry(-3.139149997073283) q[6];
rz(-1.0133173659499017) q[6];
ry(0.00236550837618843) q[7];
rz(0.19648609806659068) q[7];
ry(3.1411581624892118) q[8];
rz(1.394958084060549) q[8];
ry(-1.5546049507180717) q[9];
rz(1.5726481402561168) q[9];
ry(0.027435669193681456) q[10];
rz(-0.3618746897203531) q[10];
ry(1.5127166400406993) q[11];
rz(0.9116082371430383) q[11];
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
ry(-1.565446076364299) q[0];
rz(1.5620330814712184) q[0];
ry(2.68520583786909) q[1];
rz(-2.0476934535725135) q[1];
ry(-3.13089384713997) q[2];
rz(-2.683251234615876) q[2];
ry(1.1725231632046234) q[3];
rz(-0.9591563736354871) q[3];
ry(-1.6308247009995325) q[4];
rz(1.6875811036780934) q[4];
ry(1.5463854960444445) q[5];
rz(-2.658295706772238) q[5];
ry(-1.8606565314681431) q[6];
rz(0.11155757310919423) q[6];
ry(-1.63674553545217) q[7];
rz(-0.8840298316320176) q[7];
ry(-1.6006018843336642) q[8];
rz(-1.3085928279670318) q[8];
ry(-1.511335546902793) q[9];
rz(-0.011121499246446528) q[9];
ry(-1.9888083526093205) q[10];
rz(-1.5772535651231399) q[10];
ry(0.006808929661008989) q[11];
rz(0.6619614935084909) q[11];
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
ry(0.9937793192386692) q[0];
rz(-0.558813004813127) q[0];
ry(0.010510802069566517) q[1];
rz(0.3813315015721095) q[1];
ry(3.0389996406915243) q[2];
rz(-2.1778760795222256) q[2];
ry(3.140059638256865) q[3];
rz(0.3021205547614585) q[3];
ry(3.1320505711683793) q[4];
rz(-1.8832905033324976) q[4];
ry(-0.002465345541927455) q[5];
rz(0.29410757107820085) q[5];
ry(0.0017535590710986426) q[6];
rz(2.5475622926307477) q[6];
ry(3.1394207549117517) q[7];
rz(-2.7631923317478693) q[7];
ry(-3.1331880862327877) q[8];
rz(-1.887922555342353) q[8];
ry(3.140134318811269) q[9];
rz(0.5797556954451472) q[9];
ry(-1.5699387135702283) q[10];
rz(2.102760176076991) q[10];
ry(-1.566278233347199) q[11];
rz(0.007381300249031368) q[11];
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
ry(-1.7145581625132102) q[0];
rz(-0.5664042198089881) q[0];
ry(-1.7296193395372006) q[1];
rz(1.9737297179171571) q[1];
ry(2.1289110605777317) q[2];
rz(-2.066712806236326) q[2];
ry(1.4491864176971843) q[3];
rz(-1.2838015171008754) q[3];
ry(-2.0892863017740737) q[4];
rz(1.1016943869011273) q[4];
ry(2.7127807602211695) q[5];
rz(-1.0461991863713003) q[5];
ry(0.32971014350419386) q[6];
rz(-1.4934718179614357) q[6];
ry(1.4746488744735986) q[7];
rz(0.5033882982052763) q[7];
ry(-1.2386742066794936) q[8];
rz(-0.4176395819654024) q[8];
ry(-1.70560152314513) q[9];
rz(-0.8749456974505329) q[9];
ry(2.4795014811608795) q[10];
rz(1.600735270346064) q[10];
ry(2.100368215196834) q[11];
rz(0.9922568408214869) q[11];