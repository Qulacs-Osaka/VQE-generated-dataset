OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.0615774778557534) q[0];
rz(-1.5828551610466894) q[0];
ry(-1.5711828159805914) q[1];
rz(0.4538415414603989) q[1];
ry(3.141393707142427) q[2];
rz(-2.606675391590184) q[2];
ry(-1.570760330372635) q[3];
rz(2.1546543495039456) q[3];
ry(-1.5707874734768286) q[4];
rz(2.595371493603765) q[4];
ry(-1.570776337789396) q[5];
rz(1.570766586990736) q[5];
ry(1.5707370871093957) q[6];
rz(-0.8908178625070731) q[6];
ry(3.141513785687783) q[7];
rz(-1.201614498438549) q[7];
ry(-6.888908762010626e-06) q[8];
rz(0.8900848494907304) q[8];
ry(-3.128525289948167) q[9];
rz(-0.25844729543385375) q[9];
ry(-0.5045865025047155) q[10];
rz(-3.136561701354254) q[10];
ry(-0.17459096183848996) q[11];
rz(0.004938572011557341) q[11];
ry(-0.8414298191359374) q[12];
rz(-1.5747402737843406) q[12];
ry(-2.971186385041666) q[13];
rz(0.07439830879217782) q[13];
ry(3.1410070651466904) q[14];
rz(0.8539522517622745) q[14];
ry(-0.0004059976675449604) q[15];
rz(2.644572055393976) q[15];
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
ry(1.5725092375937144) q[0];
rz(0.5726889596720204) q[0];
ry(-0.023386406957745187) q[1];
rz(-2.076941159807644) q[1];
ry(2.833011800558837) q[2];
rz(2.5827550753678925) q[2];
ry(3.1413537677732855) q[3];
rz(0.5938836695270027) q[3];
ry(-3.1278670331469747) q[4];
rz(-2.1237197600184547) q[4];
ry(1.5673622489904981) q[5];
rz(2.5672782810844303) q[5];
ry(9.937946675009357e-05) q[6];
rz(-0.9106612635220751) q[6];
ry(-1.5709234704527626) q[7];
rz(-1.5716986943175753) q[7];
ry(3.141582885622456) q[8];
rz(-2.771524963445974) q[8];
ry(-0.0005235288824447792) q[9];
rz(-0.983452289221181) q[9];
ry(1.2118959853092814) q[10];
rz(-1.5525122339885034) q[10];
ry(-1.4016317782596075) q[11];
rz(1.5778379161944949) q[11];
ry(1.5670950132531205) q[12];
rz(2.4838540950644132) q[12];
ry(-3.1237143213161196) q[13];
rz(0.08092749104919594) q[13];
ry(-1.5716065796012035) q[14];
rz(1.6439910741122297) q[14];
ry(1.570687835189398) q[15];
rz(0.03716466317329876) q[15];
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
ry(3.13902198376645) q[0];
rz(-2.5706059847769196) q[0];
ry(1.7604595286973879) q[1];
rz(1.5605071110138509) q[1];
ry(3.141455805251449) q[2];
rz(2.1686463046175906) q[2];
ry(-0.25416180229408614) q[3];
rz(-1.9987074532764115) q[3];
ry(1.5616408604216996) q[4];
rz(1.5716207086005316) q[4];
ry(0.0031750728122093013) q[5];
rz(-0.9857757825019976) q[5];
ry(2.832010759606141) q[6];
rz(-0.953305697100415) q[6];
ry(1.570550859543208) q[7];
rz(-1.5712691552855593) q[7];
ry(-1.5707170986427432) q[8];
rz(-3.1292980692315577) q[8];
ry(-0.0037688660877570612) q[9];
rz(-2.275308350054063) q[9];
ry(-3.0320525142147656) q[10];
rz(1.5268225956989394) q[10];
ry(-2.786674450653597) q[11];
rz(-1.5637730146777002) q[11];
ry(-1.5708752138408968) q[12];
rz(0.0008290447803673828) q[12];
ry(1.5880421625901056) q[13];
rz(3.0153476235939136) q[13];
ry(-3.139894088725089) q[14];
rz(2.086316700031493) q[14];
ry(-0.00582201456457021) q[15];
rz(1.4985489148169406) q[15];
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
ry(1.581216680621837) q[0];
rz(-0.0011144143467758383) q[0];
ry(1.4148383156096425) q[1];
rz(-2.11140197476596) q[1];
ry(-3.140892736166523) q[2];
rz(-1.994933491517566) q[2];
ry(2.3325567058307684e-05) q[3];
rz(-0.4827103610873288) q[3];
ry(1.7675561081629174) q[4];
rz(-1.4546986391617303) q[4];
ry(-1.5712378402765061) q[5];
rz(-1.6451505320525988) q[5];
ry(3.1415508043728275) q[6];
rz(0.8538398705158282) q[6];
ry(2.8722659948681515) q[7];
rz(-3.1412222207276033) q[7];
ry(-3.1415519311721423) q[8];
rz(0.0008924487826552842) q[8];
ry(-3.1414332754801952) q[9];
rz(1.988229707068971) q[9];
ry(-0.0010529381333055652) q[10];
rz(2.0063354954909394) q[10];
ry(-1.5711409498594258) q[11];
rz(1.5843174114661436) q[11];
ry(-1.5710150909551066) q[12];
rz(2.114073945048682) q[12];
ry(-3.1234322075641465) q[13];
rz(-1.735603399296255) q[13];
ry(-0.1670188558538367) q[14];
rz(-0.45185547240171464) q[14];
ry(-1.4901184496034965) q[15];
rz(2.3730999411080758) q[15];
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
ry(-1.5596595523422618) q[0];
rz(0.7615487775358478) q[0];
ry(3.1400256474567882) q[1];
rz(2.8862233109739406) q[1];
ry(1.5708193003892017) q[2];
rz(-3.1339553385418983) q[2];
ry(-3.1415000371143544) q[3];
rz(-2.569696293716386) q[3];
ry(1.5649723641116768) q[4];
rz(-3.1372087908371094) q[4];
ry(3.137761579253784) q[5];
rz(1.4963599900636424) q[5];
ry(-1.5681293994463124) q[6];
rz(0.9449904795932342) q[6];
ry(-1.8371421551277853) q[7];
rz(-0.0009244829098810807) q[7];
ry(-1.5120712535036491) q[8];
rz(1.6903125346839145) q[8];
ry(-3.141185779450087) q[9];
rz(2.410579557685648) q[9];
ry(0.007652597565237507) q[10];
rz(-1.9423440083762733) q[10];
ry(-2.9029413744599917) q[11];
rz(-1.3404144899206432) q[11];
ry(9.301581388803274e-05) q[12];
rz(-1.970630705499549) q[12];
ry(-1.3869051763516586e-05) q[13];
rz(0.030049738285466535) q[13];
ry(3.0802546880969794) q[14];
rz(1.2891357614388745) q[14];
ry(8.499037630915751e-05) q[15];
rz(-3.0249851098078437) q[15];
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
ry(1.026765229400504) q[0];
rz(-1.5789016045475117) q[0];
ry(1.0233599558012305) q[1];
rz(1.707525713174806) q[1];
ry(-0.2171911643041222) q[2];
rz(-0.2017057158273412) q[2];
ry(-0.05530957319936203) q[3];
rz(-1.6354658370108588) q[3];
ry(-1.5711185293589534) q[4];
rz(1.4430886237925755) q[4];
ry(1.5709060120038094) q[5];
rz(1.5706990135654684) q[5];
ry(-3.14139807420179) q[6];
rz(-0.6264534686276724) q[6];
ry(-0.1895974527285452) q[7];
rz(-0.8596565850791622) q[7];
ry(-3.141580590353112) q[8];
rz(2.0359801026687254) q[8];
ry(-2.8719408378147095) q[9];
rz(3.141270936095955) q[9];
ry(1.5707833833219653) q[10];
rz(-0.001706518978659588) q[10];
ry(-3.1415328971247054) q[11];
rz(-2.820224347131051) q[11];
ry(-3.139550414840278) q[12];
rz(0.27444767183027374) q[12];
ry(-1.5816598148098482) q[13];
rz(0.5667724318080856) q[13];
ry(-2.6361158569794405) q[14];
rz(-2.7394029175517196) q[14];
ry(0.803542900631922) q[15];
rz(1.9054530825097933) q[15];
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
ry(-0.0007680993368524306) q[0];
rz(-1.5635979641667452) q[0];
ry(0.0003363903220350295) q[1];
rz(2.679382400456385) q[1];
ry(-2.373626908358517e-05) q[2];
rz(0.1833463696143752) q[2];
ry(1.3545314642310954e-05) q[3];
rz(-0.715840179266407) q[3];
ry(-0.004096627856809754) q[4];
rz(-2.9999284676821434) q[4];
ry(-1.6545518715783512) q[5];
rz(3.141578062508285) q[5];
ry(-1.525686642937919) q[6];
rz(0.003650462610538696) q[6];
ry(0.05450594766945205) q[7];
rz(-0.7107580034399108) q[7];
ry(3.140801162628981) q[8];
rz(0.19246628856692793) q[8];
ry(-1.5702583644153219) q[9];
rz(0.31284210663861245) q[9];
ry(2.9391214377201815) q[10];
rz(-1.572589166220733) q[10];
ry(-0.01821434952692602) q[11];
rz(1.466041599208613) q[11];
ry(0.0001661308882239254) q[12];
rz(1.44539187952312) q[12];
ry(-3.141194440274186) q[13];
rz(0.5672136441235729) q[13];
ry(3.1409265315042365) q[14];
rz(-2.689876735404346) q[14];
ry(-3.1267388180993474) q[15];
rz(3.013179174009812) q[15];
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
ry(-0.4034234772868244) q[0];
rz(-1.8836971809631686) q[0];
ry(2.997262093449376) q[1];
rz(-0.7597572313964306) q[1];
ry(3.116143729552914) q[2];
rz(2.382334239241186) q[2];
ry(-3.138461685703807) q[3];
rz(0.3713228782272381) q[3];
ry(3.1364818958437555) q[4];
rz(-1.5568287519536943) q[4];
ry(1.570020443254112) q[5];
rz(-1.5711498556343686) q[5];
ry(1.5709236412567051) q[6];
rz(3.0172372111003654e-05) q[6];
ry(-1.5708092629545451) q[7];
rz(-5.668553802725638e-06) q[7];
ry(3.133706579300712) q[8];
rz(-2.871992769892899) q[8];
ry(0.004527641687722743) q[9];
rz(2.3570999865889264) q[9];
ry(-1.57108469202364) q[10];
rz(2.8230458941994208) q[10];
ry(1.5702998142765703) q[11];
rz(-0.7108852865389664) q[11];
ry(1.5705591621496184) q[12];
rz(3.1398814858394464) q[12];
ry(-1.583112213307164) q[13];
rz(1.5620944458698904) q[13];
ry(-3.1413794143376848) q[14];
rz(-3.043841299617611) q[14];
ry(1.0919383534558271) q[15];
rz(0.39971266531932087) q[15];
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
ry(-3.1409939677576637) q[0];
rz(-0.5247054509270153) q[0];
ry(-3.1403452566459107) q[1];
rz(-0.6814502167824914) q[1];
ry(3.141514642192604) q[2];
rz(-2.315508699633871) q[2];
ry(3.141542959967415) q[3];
rz(-2.044226358231705) q[3];
ry(-1.570789007426324) q[4];
rz(-1.4280963611345472) q[4];
ry(1.5708143459979595) q[5];
rz(-1.5508063655542341) q[5];
ry(-1.570857327097703) q[6];
rz(1.6088251336363941) q[6];
ry(-1.5708007273618676) q[7];
rz(-2.1657379322251806) q[7];
ry(-3.141591939557705) q[8];
rz(-1.095399588441355) q[8];
ry(-3.141534635706727) q[9];
rz(-2.024627839792336) q[9];
ry(-3.1413522092729913) q[10];
rz(-1.9950978552658498) q[10];
ry(3.1415754113301815) q[11];
rz(0.5336546740596955) q[11];
ry(-1.5705720462222867) q[12];
rz(1.5709861785376968) q[12];
ry(-1.5705895512696981) q[13];
rz(1.5899417809609264) q[13];
ry(-3.140514610438596) q[14];
rz(0.3646533455915595) q[14];
ry(1.5708506545921583) q[15];
rz(0.39555262210819836) q[15];
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
ry(-2.5651374726059557) q[0];
rz(1.9571405045614694) q[0];
ry(1.077370794447777) q[1];
rz(2.792626881009774) q[1];
ry(-1.5800876053992983) q[2];
rz(2.2083717285084283) q[2];
ry(-1.5926770850602079) q[3];
rz(0.636890756890871) q[3];
ry(3.1377512202271354) q[4];
rz(2.3485298499050984) q[4];
ry(1.5360495635762028) q[5];
rz(0.6429034152713645) q[5];
ry(-1.5762575225221909) q[6];
rz(2.2131572236115478) q[6];
ry(-0.04088815432237008) q[7];
rz(1.2391265607272586) q[7];
ry(-1.5731908252414535) q[8];
rz(2.213112009038455) q[8];
ry(1.5302328745084086) q[9];
rz(0.6435942342508181) q[9];
ry(-0.05433250515847505) q[10];
rz(0.748348481419388) q[10];
ry(0.05421314957713896) q[11];
rz(-2.1717428764208933) q[11];
ry(-1.9606329700181016) q[12];
rz(0.640692020087568) q[12];
ry(1.1806632082350719) q[13];
rz(0.6376794520756439) q[13];
ry(0.38629068981363573) q[14];
rz(-2.501706253628283) q[14];
ry(1.5883341704745817) q[15];
rz(-0.9258742803205156) q[15];