OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-1.5772618883771026) q[0];
rz(-1.9559792158334153) q[0];
ry(-1.5548128461255235) q[1];
rz(-2.398585758820855) q[1];
ry(-0.004356254197153575) q[2];
rz(1.4919433470527892) q[2];
ry(3.139357514333852) q[3];
rz(2.6516232526147006) q[3];
ry(-0.7600544024300095) q[4];
rz(1.4591110247409258) q[4];
ry(-2.456422493257628) q[5];
rz(-1.273903265515499) q[5];
ry(-1.1300576154346595) q[6];
rz(-0.44511137439538384) q[6];
ry(0.9615435762834323) q[7];
rz(-1.287058353865846) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.3547078934623373) q[0];
rz(0.8952728980633617) q[0];
ry(-1.8258292368726645) q[1];
rz(2.5630752943069455) q[1];
ry(0.2638285854471666) q[2];
rz(-3.0188004358566674) q[2];
ry(-0.2927059363956266) q[3];
rz(2.1752630419743775) q[3];
ry(-1.6277118588605868) q[4];
rz(1.9860826131946945) q[4];
ry(-0.7200214052139632) q[5];
rz(2.396026165271046) q[5];
ry(-0.6653118996021732) q[6];
rz(-1.6758625101796263) q[6];
ry(-1.4673066054174004) q[7];
rz(0.14701251395021941) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.02608032410311776) q[0];
rz(2.514716902257011) q[0];
ry(0.011640835600076116) q[1];
rz(-2.7484549093250017) q[1];
ry(-1.4696396777568883) q[2];
rz(2.733485594061087) q[2];
ry(-0.5863988032125933) q[3];
rz(-1.2526064504005765) q[3];
ry(-2.4015366010138504) q[4];
rz(2.353515905037107) q[4];
ry(2.3217459090436736) q[5];
rz(2.3079637561751425) q[5];
ry(2.060803586697848) q[6];
rz(0.39124012522686574) q[6];
ry(-2.1997707359816094) q[7];
rz(0.9494119393968125) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.579363899910357) q[0];
rz(2.6476947971981875) q[0];
ry(-0.0035541535406906637) q[1];
rz(2.77540333669944) q[1];
ry(2.4865711687469894) q[2];
rz(2.5133459509613743) q[2];
ry(1.7052984808718321) q[3];
rz(0.9520066736027825) q[3];
ry(-3.1125140535636984) q[4];
rz(1.960299586373861) q[4];
ry(-1.5291253910576321) q[5];
rz(-0.5762462733074615) q[5];
ry(-2.211581741502593) q[6];
rz(2.0183244746275695) q[6];
ry(-2.089469247888257) q[7];
rz(-2.308055162903774) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.1362203902743198) q[0];
rz(0.4901284492502311) q[0];
ry(0.8450646103596053) q[1];
rz(-0.8420867895748665) q[1];
ry(0.012834064868250974) q[2];
rz(0.48059047843026015) q[2];
ry(3.1363320016279754) q[3];
rz(0.44384799748317505) q[3];
ry(-1.430676846971008) q[4];
rz(0.07462123863591774) q[4];
ry(-2.40292630824265) q[5];
rz(-3.0041637165756003) q[5];
ry(-1.8039120430766697) q[6];
rz(0.13508182541729052) q[6];
ry(-2.243986203141988) q[7];
rz(-0.5453296221385983) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.8197183199862642) q[0];
rz(2.483455450588415) q[0];
ry(-1.9921576424842242) q[1];
rz(-1.0866789829838615) q[1];
ry(-2.353228144291343) q[2];
rz(-1.115334413557247) q[2];
ry(-1.7781692303211019) q[3];
rz(1.7936473392461634) q[3];
ry(-0.0881023795187335) q[4];
rz(0.806853714771032) q[4];
ry(-1.1518322458214556) q[5];
rz(-0.5008509558888345) q[5];
ry(-2.4962928281563874) q[6];
rz(2.577610407045216) q[6];
ry(0.5582308914107631) q[7];
rz(1.744191117979779) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5580133759778327) q[0];
rz(3.135668935957588) q[0];
ry(-1.5637523011688244) q[1];
rz(2.701907356375945) q[1];
ry(2.0164895265121876) q[2];
rz(-0.18992468797758424) q[2];
ry(0.2773929533099251) q[3];
rz(-2.540044167854467) q[3];
ry(1.5113096718573882) q[4];
rz(0.1546632236640402) q[4];
ry(-2.567405069790692) q[5];
rz(-1.7961913929283675) q[5];
ry(0.5486574310732824) q[6];
rz(1.8816771416119256) q[6];
ry(0.9463935261135118) q[7];
rz(1.5542211155905437) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7019501247382198) q[0];
rz(-0.24796513216434415) q[0];
ry(-3.140957671493625) q[1];
rz(2.365954080018422) q[1];
ry(0.00921048174912098) q[2];
rz(-2.136852401221728) q[2];
ry(-0.0033339021119968316) q[3];
rz(-1.612146201625269) q[3];
ry(2.22169190313538) q[4];
rz(0.39699285998245326) q[4];
ry(-0.7066704756131186) q[5];
rz(-2.604616622865902) q[5];
ry(-0.6437431251909054) q[6];
rz(2.241131745921095) q[6];
ry(0.45158623552541755) q[7];
rz(-0.8591089950633551) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.8840081114841962) q[0];
rz(1.4379874563858541) q[0];
ry(1.8025111798979934) q[1];
rz(0.03894724942334005) q[1];
ry(2.0245341805387325) q[2];
rz(-0.25669213019686676) q[2];
ry(2.204478758295738) q[3];
rz(-3.115357233251291) q[3];
ry(-2.503062143940864) q[4];
rz(2.610842566455797) q[4];
ry(0.9007201273589508) q[5];
rz(-0.4089614193863351) q[5];
ry(-1.893944502765751) q[6];
rz(2.748323009543278) q[6];
ry(-0.5886378222138956) q[7];
rz(-1.971390312821797) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.227761491690966) q[0];
rz(1.3536401758524867) q[0];
ry(1.9165337651106027) q[1];
rz(1.3510500586982568) q[1];
ry(1.0243414648864118) q[2];
rz(-1.8144204159672974) q[2];
ry(2.0330137705355256) q[3];
rz(1.410719681951661) q[3];
ry(-1.7693765354707196) q[4];
rz(0.4980658748488266) q[4];
ry(-0.25891551303388266) q[5];
rz(-0.6655713547501748) q[5];
ry(-2.7801936571494625) q[6];
rz(2.0610566743186345) q[6];
ry(1.630329284283249) q[7];
rz(-0.8157362900070239) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.1200562047508613) q[0];
rz(-0.19088846918178515) q[0];
ry(-1.1121230856277302) q[1];
rz(-0.1982267469154153) q[1];
ry(-1.4962305432924305) q[2];
rz(0.25676389576763087) q[2];
ry(-1.7476873774036026) q[3];
rz(-1.7311782677898253) q[3];
ry(0.9166814829285848) q[4];
rz(2.795983639421761) q[4];
ry(2.3583812071648125) q[5];
rz(-0.4605072642844368) q[5];
ry(-2.6740806388963314) q[6];
rz(0.8935483042419774) q[6];
ry(-0.7218267951962801) q[7];
rz(1.0076572851762735) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.820958978665783) q[0];
rz(-2.0402612495443226) q[0];
ry(-0.82580287202003) q[1];
rz(2.6823029512665175) q[1];
ry(-0.6086556182579449) q[2];
rz(-1.868610478227489) q[2];
ry(2.6299646802194783) q[3];
rz(-0.8718193311651361) q[3];
ry(-2.314891352780743) q[4];
rz(-2.3732327367064845) q[4];
ry(0.9755007490445786) q[5];
rz(-1.0810047918037828) q[5];
ry(-0.3802619451132519) q[6];
rz(2.8757932366125303) q[6];
ry(-0.8576041981667286) q[7];
rz(1.3365279202999465) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.1808071626805294) q[0];
rz(0.6350537401401564) q[0];
ry(2.225307749365003) q[1];
rz(0.307687423394488) q[1];
ry(1.6269683040595628) q[2];
rz(-2.0365464554797983) q[2];
ry(2.415463350388314) q[3];
rz(-2.7501810051993214) q[3];
ry(1.9186130542840723) q[4];
rz(-0.41128832832781725) q[4];
ry(-0.1268327371136954) q[5];
rz(0.9961235085180589) q[5];
ry(2.177702285805113) q[6];
rz(0.6037361419590185) q[6];
ry(-2.5322143508989776) q[7];
rz(0.18864627553739993) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.3765235220228216) q[0];
rz(1.5934084073382326) q[0];
ry(-2.724358550249786) q[1];
rz(-0.039410614920427633) q[1];
ry(-0.007271744017147571) q[2];
rz(-1.8978043391125947) q[2];
ry(0.0011785963961932293) q[3];
rz(-0.49982318747616317) q[3];
ry(-2.9646219971123666) q[4];
rz(0.3665045116905211) q[4];
ry(-1.8159284773534607) q[5];
rz(2.4643624892222857) q[5];
ry(-2.9772774451675788) q[6];
rz(-1.26714174788914) q[6];
ry(-2.901788624124637) q[7];
rz(1.188946718540505) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.3566817073284176) q[0];
rz(-2.3070275163824134) q[0];
ry(0.011434819443865754) q[1];
rz(-2.4689700478127983) q[1];
ry(0.16505993804820926) q[2];
rz(-1.7376944802528254) q[2];
ry(3.1074606588928475) q[3];
rz(1.7933064455874845) q[3];
ry(2.6249857842958484) q[4];
rz(-1.486452940888074) q[4];
ry(-1.9177786049450196) q[5];
rz(-0.8041845211438066) q[5];
ry(0.896689653996771) q[6];
rz(-2.4416863092853545) q[6];
ry(1.1001287985525616) q[7];
rz(0.28141073478494044) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.8119772458711574) q[0];
rz(-2.5653986487403344) q[0];
ry(0.39809744110827094) q[1];
rz(-2.440633231093157) q[1];
ry(-1.856653710790355) q[2];
rz(-0.3285466082865867) q[2];
ry(1.4990623399128884) q[3];
rz(1.6056101156911475) q[3];
ry(-2.5821417537956406) q[4];
rz(-2.8024108034020583) q[4];
ry(-1.8047380068040058) q[5];
rz(-2.9756336160765593) q[5];
ry(1.825140705343923) q[6];
rz(2.4655004466690453) q[6];
ry(-2.8088932916795297) q[7];
rz(-0.8642500668934474) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.404846705335836) q[0];
rz(-1.039953778301176) q[0];
ry(0.4323802738771443) q[1];
rz(2.0585728658875815) q[1];
ry(0.15597162275778728) q[2];
rz(1.1788390274722629) q[2];
ry(0.712530451267375) q[3];
rz(1.2731117983145726) q[3];
ry(2.051271919505628) q[4];
rz(-2.2668236197448564) q[4];
ry(0.611827457082743) q[5];
rz(1.382760590295791) q[5];
ry(-2.0066113497813736) q[6];
rz(-2.6246132608628656) q[6];
ry(1.9776956506805288) q[7];
rz(1.1282060477755298) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.8108558435241413) q[0];
rz(3.00520734655604) q[0];
ry(0.33540761016565046) q[1];
rz(0.19139093983170774) q[1];
ry(-0.07903620734174677) q[2];
rz(2.3571079715710925) q[2];
ry(-0.3525124593259737) q[3];
rz(2.1146230358337057) q[3];
ry(-1.6113224140045939) q[4];
rz(0.5039591896457892) q[4];
ry(-2.077401968780945) q[5];
rz(-0.49369240362327155) q[5];
ry(-2.501988919600602) q[6];
rz(1.4718541546824913) q[6];
ry(-2.622835946408901) q[7];
rz(2.9786537261600334) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.317754961613238) q[0];
rz(-0.6475931185275989) q[0];
ry(-0.8339380847222913) q[1];
rz(0.954591318869183) q[1];
ry(-1.2517688540747889) q[2];
rz(-1.1727142307704472) q[2];
ry(-2.1041763822826587) q[3];
rz(-2.756355528003723) q[3];
ry(0.5279694388511613) q[4];
rz(0.2562808546782093) q[4];
ry(-2.1921644418011654) q[5];
rz(0.19506914719607238) q[5];
ry(-0.25534701915740493) q[6];
rz(2.170176289277051) q[6];
ry(-2.7021491234918344) q[7];
rz(1.0113453359020808) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.334008781832007) q[0];
rz(-2.650242272375539) q[0];
ry(-1.5918009929060977) q[1];
rz(-0.36806086653580383) q[1];
ry(3.1185502851463553) q[2];
rz(1.6992001423880811) q[2];
ry(1.9077505138355813) q[3];
rz(-0.6538533404756777) q[3];
ry(2.0889166833857034) q[4];
rz(-1.752094526012271) q[4];
ry(-0.16322081248342232) q[5];
rz(-0.6649537210379963) q[5];
ry(2.925526826739134) q[6];
rz(0.6171742571094121) q[6];
ry(-1.3247918795390081) q[7];
rz(0.5714148530729686) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.33979196673806555) q[0];
rz(-0.7912551591294614) q[0];
ry(-2.7372735211070545) q[1];
rz(0.3376360148906583) q[1];
ry(-3.1373858195111266) q[2];
rz(-0.46907033084150956) q[2];
ry(-3.1405420067167507) q[3];
rz(-0.028699682241047988) q[3];
ry(2.1671179591100054) q[4];
rz(-1.3153480551942673) q[4];
ry(1.12500532818758) q[5];
rz(3.029574345784466) q[5];
ry(-1.5870530089142019) q[6];
rz(-1.890014497883815) q[6];
ry(-2.2220984112204345) q[7];
rz(2.9617644724400507) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.0283655749680767) q[0];
rz(-1.1176807840577556) q[0];
ry(-2.6628700192411046) q[1];
rz(-2.0623358875417623) q[1];
ry(1.5620059135174857) q[2];
rz(1.3833707490943357) q[2];
ry(-1.5633360870324648) q[3];
rz(-1.5617406560750833) q[3];
ry(-1.5260109090730465) q[4];
rz(0.36270117388717305) q[4];
ry(-1.6975188986088128) q[5];
rz(-1.5447400024043685) q[5];
ry(1.639271424638741) q[6];
rz(-2.914345246372355) q[6];
ry(-1.2126576913272984) q[7];
rz(-0.49464980343773834) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.2445886026266393) q[0];
rz(1.743081173093933) q[0];
ry(0.010003054175623305) q[1];
rz(-2.2613038852373206) q[1];
ry(-1.806991292965948) q[2];
rz(1.0043369139923786) q[2];
ry(-1.5921853690268943) q[3];
rz(-0.9512280667049474) q[3];
ry(-1.6425490037865949) q[4];
rz(0.37551116323861783) q[4];
ry(-0.2345361027583328) q[5];
rz(0.015294833816785041) q[5];
ry(-2.0405270892636294) q[6];
rz(-1.2090637598596754) q[6];
ry(-1.7735727224999185) q[7];
rz(1.9986669037802725) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4224827536782128) q[0];
rz(-1.5759995900366095) q[0];
ry(2.8543917224349613) q[1];
rz(-0.4924001701175675) q[1];
ry(0.0030920523251465326) q[2];
rz(-2.6558257586264298) q[2];
ry(-3.1362287779231863) q[3];
rz(-2.4552345140290504) q[3];
ry(-0.0033490696627849204) q[4];
rz(0.1430123827158729) q[4];
ry(-0.00589003614397221) q[5];
rz(-3.1270537675440804) q[5];
ry(-0.6427800611723147) q[6];
rz(0.6693862038937474) q[6];
ry(2.9735643937775356) q[7];
rz(-2.9273542767458474) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5556613638569274) q[0];
rz(-0.5568448208592596) q[0];
ry(-0.00641259774803787) q[1];
rz(0.4826810561294572) q[1];
ry(-1.5312989687166192) q[2];
rz(-0.6060802839180442) q[2];
ry(-1.5608536524216907) q[3];
rz(-3.109551048175599) q[3];
ry(2.0544341116733653) q[4];
rz(1.0852355039627986) q[4];
ry(0.5833289401089186) q[5];
rz(0.06820087811008424) q[5];
ry(1.9021692873262852) q[6];
rz(-0.8211730585849308) q[6];
ry(-2.2606066607777686) q[7];
rz(1.8043191402428453) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5396021873187715) q[0];
rz(2.987448407086724) q[0];
ry(-1.5860802031835515) q[1];
rz(-2.9568811270037556) q[1];
ry(3.122775164846388) q[2];
rz(-2.4711845826160226) q[2];
ry(-0.032619003702458134) q[3];
rz(3.031013286589658) q[3];
ry(-0.1532652815382298) q[4];
rz(1.9817842392382197) q[4];
ry(-0.09866156819685264) q[5];
rz(1.4596657827247463) q[5];
ry(-3.000208110150476) q[6];
rz(-2.391263881320965) q[6];
ry(2.192580236470026) q[7];
rz(-2.0751678271839697) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.017298767210168542) q[0];
rz(-3.0358843964114857) q[0];
ry(0.11814506152432422) q[1];
rz(-0.8013521958119205) q[1];
ry(0.0016964087512224126) q[2];
rz(1.989545013516322) q[2];
ry(0.0014069141823274123) q[3];
rz(0.06057450485973704) q[3];
ry(-1.9793105288812731) q[4];
rz(1.9736339179216076) q[4];
ry(2.0347745378685005) q[5];
rz(1.9511097132368338) q[5];
ry(0.053963043032873984) q[6];
rz(0.6759179168720871) q[6];
ry(-1.5848054163596261) q[7];
rz(-0.27525224733938103) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5439980463799072) q[0];
rz(-3.1040282057852155) q[0];
ry(0.004760835817754311) q[1];
rz(2.2450148088943482) q[1];
ry(-3.1235045987669614) q[2];
rz(0.4035284294901897) q[2];
ry(1.3368866071709853) q[3];
rz(1.6185989593587182) q[3];
ry(-1.7557258001064604) q[4];
rz(-1.2817626906203359) q[4];
ry(1.5868222815922206) q[5];
rz(-3.109422387362645) q[5];
ry(1.493839269720124) q[6];
rz(-0.043192555206969546) q[6];
ry(-2.6732446951711037) q[7];
rz(1.5340477244653803) q[7];