OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.044942231555069156) q[0];
rz(1.677571076153369) q[0];
ry(0.06380115993546553) q[1];
rz(1.6456500637694396) q[1];
ry(-0.5432948225260271) q[2];
rz(-1.3649685662504663) q[2];
ry(0.26840577131576904) q[3];
rz(-0.9329175960447306) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.0926516304622633) q[0];
rz(-1.7127509232999607) q[0];
ry(-1.0770113700805704) q[1];
rz(-2.7314580641209805) q[1];
ry(-0.4372491008496695) q[2];
rz(-0.372711732005595) q[2];
ry(-0.9358705398357278) q[3];
rz(0.9981513684304921) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.40146508931880487) q[0];
rz(0.6335498660665323) q[0];
ry(0.9058698509144267) q[1];
rz(1.1486870678290446) q[1];
ry(-1.5270294704606477) q[2];
rz(-2.300013103278492) q[2];
ry(1.5424630093407785) q[3];
rz(1.2490903323128688) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.7180716139269361) q[0];
rz(2.865048813753582) q[0];
ry(1.07616907056679) q[1];
rz(2.4456234416647527) q[1];
ry(1.3390538984792917) q[2];
rz(2.6118185191249514) q[2];
ry(0.4891620413663012) q[3];
rz(0.24445657024582593) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.8958197503704177) q[0];
rz(0.8615043807916524) q[0];
ry(0.020636215723354813) q[1];
rz(-2.2346776158362047) q[1];
ry(-0.7420499464902548) q[2];
rz(0.9322488605696158) q[2];
ry(1.3106804286879163) q[3];
rz(0.5248403590079453) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.426364372352328) q[0];
rz(0.7970893905615624) q[0];
ry(2.737624585008823) q[1];
rz(-1.31281836152064) q[1];
ry(0.1361887409187137) q[2];
rz(1.3218583265411739) q[2];
ry(2.9836967346102727) q[3];
rz(3.1373036936359844) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.23237989057260428) q[0];
rz(0.38123094595431173) q[0];
ry(2.6694729138426516) q[1];
rz(2.4448305642723627) q[1];
ry(-0.8187274152657169) q[2];
rz(-2.235774074460415) q[2];
ry(-0.17258553005149135) q[3];
rz(1.4014860245953518) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8206321767126195) q[0];
rz(0.9584889994567306) q[0];
ry(-0.684410300567242) q[1];
rz(-1.8727492952824611) q[1];
ry(-1.3624871113179085) q[2];
rz(-1.2020689337905548) q[2];
ry(-0.6777229028595864) q[3];
rz(-1.3567471865938614) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.3666802933678444) q[0];
rz(0.27015889482124145) q[0];
ry(-1.3435868324173272) q[1];
rz(-1.1831755688405228) q[1];
ry(0.7958242119368766) q[2];
rz(2.173088215007521) q[2];
ry(1.668818500223305) q[3];
rz(0.8145910916633872) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.33278247033709313) q[0];
rz(-1.2268004300463782) q[0];
ry(2.1510788328482175) q[1];
rz(-2.825611657491233) q[1];
ry(2.2414681083903036) q[2];
rz(2.0962466537468276) q[2];
ry(-0.7327783257115925) q[3];
rz(0.3051238644814554) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.4301543119915685) q[0];
rz(-1.4335212535204533) q[0];
ry(2.41047297833115) q[1];
rz(-2.5610630404401347) q[1];
ry(-1.1526164332158948) q[2];
rz(1.950854371600661) q[2];
ry(0.3626762421542172) q[3];
rz(1.8353223321648517) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.6004104136731974) q[0];
rz(2.8136272034942214) q[0];
ry(-2.337665063830172) q[1];
rz(2.5004938262932135) q[1];
ry(-1.5774149507592499) q[2];
rz(1.155136050995703) q[2];
ry(0.11789782695013332) q[3];
rz(1.622600402902366) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.9870796463354232) q[0];
rz(2.4920822545177033) q[0];
ry(-0.5418659995537844) q[1];
rz(-1.9156665777231077) q[1];
ry(0.16490539058932635) q[2];
rz(2.8879969538921455) q[2];
ry(-2.7684714899890914) q[3];
rz(1.5907852466808388) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.3274860557890422) q[0];
rz(-1.311032298828595) q[0];
ry(0.6655448752396874) q[1];
rz(-0.7667026045269275) q[1];
ry(2.997172958824993) q[2];
rz(1.0987788547716824) q[2];
ry(0.14227856419611662) q[3];
rz(-0.39008931908348027) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.742653162429668) q[0];
rz(-0.09088673733232228) q[0];
ry(0.45741038144826357) q[1];
rz(2.471367546050596) q[1];
ry(2.825762800086711) q[2];
rz(-0.724436884436242) q[2];
ry(1.3385452737930619) q[3];
rz(1.6451621924187867) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.5807087243010303) q[0];
rz(-0.8546988770571256) q[0];
ry(-1.7679141740548991) q[1];
rz(-1.1843373663807162) q[1];
ry(1.8353861454870604) q[2];
rz(2.888501505813212) q[2];
ry(-0.4208536900678169) q[3];
rz(2.991236524398748) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.7767580792544364) q[0];
rz(-1.3617295666898628) q[0];
ry(1.5852523059581523) q[1];
rz(-1.6676732363976599) q[1];
ry(0.8510760196923446) q[2];
rz(-1.854591211993352) q[2];
ry(-1.5190081835522578) q[3];
rz(-1.7577461677022592) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.8029112074536054) q[0];
rz(0.5830238194796581) q[0];
ry(1.7775936834299753) q[1];
rz(1.2630855828016727) q[1];
ry(-1.5305060430412514) q[2];
rz(0.539001501392107) q[2];
ry(-0.41891353604351167) q[3];
rz(-1.6001024816058473) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.1576608226028533) q[0];
rz(0.1279199879298881) q[0];
ry(3.0005601343323844) q[1];
rz(-1.9814288409679994) q[1];
ry(2.95064442514731) q[2];
rz(0.43187670976250736) q[2];
ry(1.5402211045428533) q[3];
rz(-2.3443450946881943) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.8813683313323146) q[0];
rz(2.632662548079992) q[0];
ry(0.5650688725203975) q[1];
rz(1.1595115665461388) q[1];
ry(-0.8415200261788272) q[2];
rz(-2.0912551682921183) q[2];
ry(-0.5630023169016853) q[3];
rz(2.4490466245988087) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.006309575592939) q[0];
rz(-0.2619186533892046) q[0];
ry(-1.7984720445294045) q[1];
rz(1.584608832118148) q[1];
ry(-0.30034148309336484) q[2];
rz(-2.5636374311307812) q[2];
ry(-2.912734562739745) q[3];
rz(0.7064629969579705) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.230240392365478) q[0];
rz(-2.5512172028086937) q[0];
ry(0.6979337254987849) q[1];
rz(-2.8643313145139118) q[1];
ry(0.27003773554843313) q[2];
rz(-0.09459686413920565) q[2];
ry(0.8611077294775705) q[3];
rz(0.10257128542859646) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.5258131656573468) q[0];
rz(1.6504295594528946) q[0];
ry(2.380813857787269) q[1];
rz(-1.3649318699638666) q[1];
ry(-2.061765686114857) q[2];
rz(1.8746512629583556) q[2];
ry(3.017232308296648) q[3];
rz(-1.8669831441098794) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6171846561720518) q[0];
rz(1.7611273585979994) q[0];
ry(-1.5247442148999923) q[1];
rz(-0.34709168704536486) q[1];
ry(-3.1325388854756593) q[2];
rz(1.3976842597103074) q[2];
ry(2.358883457078308) q[3];
rz(2.789877950769831) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.0918000854274485) q[0];
rz(-2.014977724137652) q[0];
ry(0.08590828581917885) q[1];
rz(2.327199795765162) q[1];
ry(1.6267806561892675) q[2];
rz(1.7988166203886788) q[2];
ry(-1.0025140037414682) q[3];
rz(-1.8520484278558886) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.28942682953756815) q[0];
rz(-2.501992552146718) q[0];
ry(-0.5523378494076185) q[1];
rz(0.44002572514236077) q[1];
ry(-1.1022341162010854) q[2];
rz(3.0852885239319185) q[2];
ry(-1.355588826243495) q[3];
rz(1.0642405567772206) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.967543385710231) q[0];
rz(1.9180963422383286) q[0];
ry(0.8084549789963509) q[1];
rz(-1.6301746084756896) q[1];
ry(0.3747582604046511) q[2];
rz(-0.6794083012886264) q[2];
ry(-2.058997905323947) q[3];
rz(-2.7769037909460494) q[3];