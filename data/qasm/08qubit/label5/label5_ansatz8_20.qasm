OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.3046398441188227) q[0];
ry(-2.1139862801320177) q[1];
cx q[0],q[1];
ry(-2.220368583869694) q[0];
ry(-2.486694950021879) q[1];
cx q[0],q[1];
ry(1.0210534465534762) q[2];
ry(2.0483711404108487) q[3];
cx q[2],q[3];
ry(2.5700174952327166) q[2];
ry(0.15396318286390237) q[3];
cx q[2],q[3];
ry(-2.1126640676679367) q[4];
ry(-1.5044251560052992) q[5];
cx q[4],q[5];
ry(0.37336844631470767) q[4];
ry(1.447804408575827) q[5];
cx q[4],q[5];
ry(-0.6589545862209739) q[6];
ry(2.508650488405877) q[7];
cx q[6],q[7];
ry(-0.06030899485425367) q[6];
ry(-2.898770688396375) q[7];
cx q[6],q[7];
ry(0.18775144845788017) q[0];
ry(2.0160546645916915) q[2];
cx q[0],q[2];
ry(-2.433101742540508) q[0];
ry(-2.296967318386608) q[2];
cx q[0],q[2];
ry(0.389597438526493) q[2];
ry(-1.0218850047264798) q[4];
cx q[2],q[4];
ry(0.0467801799175587) q[2];
ry(-0.8080806971119828) q[4];
cx q[2],q[4];
ry(2.446112741308466) q[4];
ry(0.429385996558921) q[6];
cx q[4],q[6];
ry(2.421608868088317) q[4];
ry(0.8523539929543967) q[6];
cx q[4],q[6];
ry(2.326727386745259) q[1];
ry(-2.877649084088372) q[3];
cx q[1],q[3];
ry(1.5497784171871791) q[1];
ry(-1.7324633171557096) q[3];
cx q[1],q[3];
ry(2.2677022031107303) q[3];
ry(-1.745624798723366) q[5];
cx q[3],q[5];
ry(-1.5667193487495403) q[3];
ry(1.578462133056255) q[5];
cx q[3],q[5];
ry(-0.17779662487753534) q[5];
ry(2.423523919709494) q[7];
cx q[5],q[7];
ry(-1.5063316680498362) q[5];
ry(3.106282448672113) q[7];
cx q[5],q[7];
ry(0.3233400552573409) q[0];
ry(0.18499868738614636) q[1];
cx q[0],q[1];
ry(-1.6758057004629479) q[0];
ry(-0.3054072823429106) q[1];
cx q[0],q[1];
ry(0.41917654050588204) q[2];
ry(2.4723989184399175) q[3];
cx q[2],q[3];
ry(-2.0449503142849346) q[2];
ry(0.8445640674187738) q[3];
cx q[2],q[3];
ry(-0.1463532182335) q[4];
ry(0.7681557718470637) q[5];
cx q[4],q[5];
ry(2.7152417307522456) q[4];
ry(0.8016267891118555) q[5];
cx q[4],q[5];
ry(1.6134169794950093) q[6];
ry(2.10178210442634) q[7];
cx q[6],q[7];
ry(-2.1730014140843767) q[6];
ry(0.5934994165200447) q[7];
cx q[6],q[7];
ry(-0.7136643957371307) q[0];
ry(0.8710566083714317) q[2];
cx q[0],q[2];
ry(-0.5565346893943093) q[0];
ry(-0.8982516869269928) q[2];
cx q[0],q[2];
ry(-2.7884154949346525) q[2];
ry(2.075999436015665) q[4];
cx q[2],q[4];
ry(-0.4383219142945967) q[2];
ry(1.2213660116967358) q[4];
cx q[2],q[4];
ry(-1.5249756866291542) q[4];
ry(0.5604733337363362) q[6];
cx q[4],q[6];
ry(0.07164687968340203) q[4];
ry(-1.4146450048028965) q[6];
cx q[4],q[6];
ry(1.1493985831743707) q[1];
ry(1.0894330021570555) q[3];
cx q[1],q[3];
ry(-0.35933657733591406) q[1];
ry(-1.7470171083067818) q[3];
cx q[1],q[3];
ry(-1.3480113368364828) q[3];
ry(2.543376062331003) q[5];
cx q[3],q[5];
ry(1.7476599291691666) q[3];
ry(2.799542237668455) q[5];
cx q[3],q[5];
ry(-0.5753189094017221) q[5];
ry(2.0574254065420607) q[7];
cx q[5],q[7];
ry(3.1370233624164023) q[5];
ry(0.7093157128628349) q[7];
cx q[5],q[7];
ry(-1.7534000862314798) q[0];
ry(-1.0547530417856812) q[1];
cx q[0],q[1];
ry(0.7038628013432825) q[0];
ry(-0.056419978236024626) q[1];
cx q[0],q[1];
ry(1.7747849201114843) q[2];
ry(1.275698647480893) q[3];
cx q[2],q[3];
ry(3.11809002632689) q[2];
ry(2.8435896952203117) q[3];
cx q[2],q[3];
ry(-2.7478162569936355) q[4];
ry(2.418599703435703) q[5];
cx q[4],q[5];
ry(0.21545151418796585) q[4];
ry(2.666202874219162) q[5];
cx q[4],q[5];
ry(1.761151966714013) q[6];
ry(-2.7637826991549774) q[7];
cx q[6],q[7];
ry(1.928940731155617) q[6];
ry(-3.0635004124996805) q[7];
cx q[6],q[7];
ry(-1.6437884159931695) q[0];
ry(2.716839067328634) q[2];
cx q[0],q[2];
ry(0.9209194552352659) q[0];
ry(1.6270106845102896) q[2];
cx q[0],q[2];
ry(-0.8605455979337453) q[2];
ry(-1.1675128289835597) q[4];
cx q[2],q[4];
ry(-1.4636757315141644) q[2];
ry(-1.7321380160657296) q[4];
cx q[2],q[4];
ry(-2.868040880265926) q[4];
ry(-2.805893148969404) q[6];
cx q[4],q[6];
ry(1.8745306242608748) q[4];
ry(0.0532215115527009) q[6];
cx q[4],q[6];
ry(-0.2125313448813289) q[1];
ry(1.169290941766198) q[3];
cx q[1],q[3];
ry(-1.74507474224356) q[1];
ry(2.9459479193359734) q[3];
cx q[1],q[3];
ry(-2.598294810461243) q[3];
ry(1.9586238863984013) q[5];
cx q[3],q[5];
ry(-2.025843716325595) q[3];
ry(-1.435047749457544) q[5];
cx q[3],q[5];
ry(1.3561907026586635) q[5];
ry(-2.822437860665827) q[7];
cx q[5],q[7];
ry(2.3392386643978194) q[5];
ry(-0.11536618170910273) q[7];
cx q[5],q[7];
ry(-3.1116199719339868) q[0];
ry(-2.9786599961774716) q[1];
cx q[0],q[1];
ry(-0.48424795259769127) q[0];
ry(1.8575493970357504) q[1];
cx q[0],q[1];
ry(2.3705702829423028) q[2];
ry(-2.922967225919364) q[3];
cx q[2],q[3];
ry(2.6334498267128326) q[2];
ry(0.09592942012084205) q[3];
cx q[2],q[3];
ry(-1.7874749189353727) q[4];
ry(2.062920656578192) q[5];
cx q[4],q[5];
ry(-2.9639399389728727) q[4];
ry(2.567778822097665) q[5];
cx q[4],q[5];
ry(0.49324754469381027) q[6];
ry(0.30908604495903713) q[7];
cx q[6],q[7];
ry(2.0050431498988446) q[6];
ry(2.6547498947337718) q[7];
cx q[6],q[7];
ry(-1.075477304381544) q[0];
ry(0.10208965792193236) q[2];
cx q[0],q[2];
ry(2.109285599474174) q[0];
ry(1.3833173009335507) q[2];
cx q[0],q[2];
ry(-1.646540283963781) q[2];
ry(-2.4176977347811905) q[4];
cx q[2],q[4];
ry(-3.1039236374980694) q[2];
ry(-0.8273390077891056) q[4];
cx q[2],q[4];
ry(0.011788750478825882) q[4];
ry(2.040075571618148) q[6];
cx q[4],q[6];
ry(1.1735195253648263) q[4];
ry(0.9766068429420718) q[6];
cx q[4],q[6];
ry(0.5989658563143341) q[1];
ry(3.1041322718701516) q[3];
cx q[1],q[3];
ry(1.826298515006307) q[1];
ry(-0.2106772065307668) q[3];
cx q[1],q[3];
ry(-0.09258745556225369) q[3];
ry(-1.4709868776902688) q[5];
cx q[3],q[5];
ry(-2.942931623618839) q[3];
ry(-1.493165050296592) q[5];
cx q[3],q[5];
ry(-2.4837505361358403) q[5];
ry(-2.3125568208556464) q[7];
cx q[5],q[7];
ry(0.7566760502319091) q[5];
ry(-1.9965578671158157) q[7];
cx q[5],q[7];
ry(-1.891827851840059) q[0];
ry(-0.527497974295665) q[1];
cx q[0],q[1];
ry(0.538225877404484) q[0];
ry(2.1814229122649405) q[1];
cx q[0],q[1];
ry(2.7528458407469114) q[2];
ry(-2.9470408794751823) q[3];
cx q[2],q[3];
ry(-1.0709199831279914) q[2];
ry(0.7871473733015462) q[3];
cx q[2],q[3];
ry(0.8130311707591638) q[4];
ry(1.2977311675023797) q[5];
cx q[4],q[5];
ry(-1.7451622176091939) q[4];
ry(1.7761070378630457) q[5];
cx q[4],q[5];
ry(-0.34847336997284284) q[6];
ry(3.0741783058212477) q[7];
cx q[6],q[7];
ry(-0.18456956457008472) q[6];
ry(-2.5873229147981163) q[7];
cx q[6],q[7];
ry(0.06379587086499319) q[0];
ry(2.932209560726606) q[2];
cx q[0],q[2];
ry(-1.3270249955935989) q[0];
ry(-2.7287461330091767) q[2];
cx q[0],q[2];
ry(1.815842769558242) q[2];
ry(-1.082617242116025) q[4];
cx q[2],q[4];
ry(1.9833392221311685) q[2];
ry(-0.03545338469860764) q[4];
cx q[2],q[4];
ry(-0.11418834822553059) q[4];
ry(-1.1224214879808687) q[6];
cx q[4],q[6];
ry(-1.7474592086602163) q[4];
ry(-1.4393405165992217) q[6];
cx q[4],q[6];
ry(-1.2947046207305029) q[1];
ry(-2.507197010635009) q[3];
cx q[1],q[3];
ry(3.06070088411383) q[1];
ry(1.0393343360675398) q[3];
cx q[1],q[3];
ry(-2.5274780648234025) q[3];
ry(1.9223141120571094) q[5];
cx q[3],q[5];
ry(-0.08673629211519152) q[3];
ry(1.5311896096776456) q[5];
cx q[3],q[5];
ry(-0.9425882555442656) q[5];
ry(0.7009543762289467) q[7];
cx q[5],q[7];
ry(0.4335233041747362) q[5];
ry(3.021498635640871) q[7];
cx q[5],q[7];
ry(-2.9835278301342494) q[0];
ry(1.2769573703717687) q[1];
cx q[0],q[1];
ry(0.4967947015278389) q[0];
ry(-1.8094500666773592) q[1];
cx q[0],q[1];
ry(-0.9196145424323083) q[2];
ry(-2.197567981560155) q[3];
cx q[2],q[3];
ry(-0.4348272610388406) q[2];
ry(-0.8108617093708034) q[3];
cx q[2],q[3];
ry(2.7767668039600744) q[4];
ry(0.9059782439732963) q[5];
cx q[4],q[5];
ry(-1.0188006097412128) q[4];
ry(2.2006771316040252) q[5];
cx q[4],q[5];
ry(-2.4054972616111643) q[6];
ry(0.28745018776979053) q[7];
cx q[6],q[7];
ry(-2.191479110545651) q[6];
ry(2.181705418702629) q[7];
cx q[6],q[7];
ry(-0.28122545014705125) q[0];
ry(0.8918901542661929) q[2];
cx q[0],q[2];
ry(0.6513081049474553) q[0];
ry(-0.6957090177318079) q[2];
cx q[0],q[2];
ry(1.6070759787731654) q[2];
ry(-2.09532413054716) q[4];
cx q[2],q[4];
ry(-2.2925607191934585) q[2];
ry(-2.981799027738338) q[4];
cx q[2],q[4];
ry(0.5272598532933988) q[4];
ry(1.3300629861758848) q[6];
cx q[4],q[6];
ry(-2.923735317958823) q[4];
ry(-2.692167531895789) q[6];
cx q[4],q[6];
ry(-1.541690804268625) q[1];
ry(1.5131406313733666) q[3];
cx q[1],q[3];
ry(-1.5218428299868634) q[1];
ry(-0.40774112934856155) q[3];
cx q[1],q[3];
ry(-2.9204624938124013) q[3];
ry(-1.9452786429191289) q[5];
cx q[3],q[5];
ry(2.04674514884086) q[3];
ry(-2.6739805440032476) q[5];
cx q[3],q[5];
ry(0.7871994324863895) q[5];
ry(-0.13735506180757895) q[7];
cx q[5],q[7];
ry(-1.4842509211239054) q[5];
ry(-2.802425887394785) q[7];
cx q[5],q[7];
ry(-0.6043919295782234) q[0];
ry(-2.7977881189619085) q[1];
cx q[0],q[1];
ry(2.7240071154553407) q[0];
ry(-0.9527858473880464) q[1];
cx q[0],q[1];
ry(0.24875208291248718) q[2];
ry(-0.2878204194815388) q[3];
cx q[2],q[3];
ry(-0.7332997920818167) q[2];
ry(0.45803191502247453) q[3];
cx q[2],q[3];
ry(-1.6593781978528552) q[4];
ry(-1.6126262553908803) q[5];
cx q[4],q[5];
ry(-1.8271120512385737) q[4];
ry(-0.9590703205268376) q[5];
cx q[4],q[5];
ry(-0.30131371340156643) q[6];
ry(-1.1128404151302433) q[7];
cx q[6],q[7];
ry(2.4579373165160043) q[6];
ry(-2.5305762140006913) q[7];
cx q[6],q[7];
ry(-0.8021143246468276) q[0];
ry(-0.28862494016386364) q[2];
cx q[0],q[2];
ry(2.956239100963717) q[0];
ry(-3.112577298888903) q[2];
cx q[0],q[2];
ry(-2.0346996720590402) q[2];
ry(-2.4311143165219695) q[4];
cx q[2],q[4];
ry(-3.1053330285192917) q[2];
ry(0.9557456497741549) q[4];
cx q[2],q[4];
ry(1.3155078204837514) q[4];
ry(1.6906494542999455) q[6];
cx q[4],q[6];
ry(-0.5399539584229762) q[4];
ry(1.48195064776305) q[6];
cx q[4],q[6];
ry(-2.3594060116146998) q[1];
ry(2.924230087148706) q[3];
cx q[1],q[3];
ry(-1.6732662065450743) q[1];
ry(-0.8285013112849686) q[3];
cx q[1],q[3];
ry(2.3238671287325112) q[3];
ry(-1.9648606107886684) q[5];
cx q[3],q[5];
ry(2.9790584830037115) q[3];
ry(-2.2833063907041184) q[5];
cx q[3],q[5];
ry(-2.582348443571272) q[5];
ry(-0.7725533022945996) q[7];
cx q[5],q[7];
ry(-2.5053528020473617) q[5];
ry(1.8661945009212293) q[7];
cx q[5],q[7];
ry(2.1846962858035437) q[0];
ry(1.1501239157457848) q[1];
cx q[0],q[1];
ry(2.993334204426266) q[0];
ry(-2.7137885029250595) q[1];
cx q[0],q[1];
ry(-0.9233212598391415) q[2];
ry(-1.5663185281730152) q[3];
cx q[2],q[3];
ry(2.1856392458369935) q[2];
ry(0.3026308000893341) q[3];
cx q[2],q[3];
ry(2.1827104655832352) q[4];
ry(2.6051211995052417) q[5];
cx q[4],q[5];
ry(-1.4957506911760223) q[4];
ry(-2.806392749032802) q[5];
cx q[4],q[5];
ry(2.570439797194417) q[6];
ry(-0.14314059491787848) q[7];
cx q[6],q[7];
ry(-0.6373669881446999) q[6];
ry(0.8119988438218575) q[7];
cx q[6],q[7];
ry(1.800476405533633) q[0];
ry(-1.6369151664160162) q[2];
cx q[0],q[2];
ry(2.8561644010783613) q[0];
ry(-1.058579383759528) q[2];
cx q[0],q[2];
ry(1.6712791132121085) q[2];
ry(-1.854307825560201) q[4];
cx q[2],q[4];
ry(-2.5168011758978728) q[2];
ry(-2.4558052466526226) q[4];
cx q[2],q[4];
ry(1.239927173612369) q[4];
ry(-2.341447569929239) q[6];
cx q[4],q[6];
ry(1.9210831123130967) q[4];
ry(-0.7277757537441732) q[6];
cx q[4],q[6];
ry(-1.3458853351060704) q[1];
ry(2.9789014276635917) q[3];
cx q[1],q[3];
ry(0.8680235748381605) q[1];
ry(2.105545012640424) q[3];
cx q[1],q[3];
ry(-1.4906306088874963) q[3];
ry(0.9998700584402321) q[5];
cx q[3],q[5];
ry(0.6084660243327704) q[3];
ry(0.650112870860938) q[5];
cx q[3],q[5];
ry(-2.0951642736367466) q[5];
ry(2.4279811338121084) q[7];
cx q[5],q[7];
ry(-0.43855060086267716) q[5];
ry(-0.35601368159066027) q[7];
cx q[5],q[7];
ry(-2.4776782544401392) q[0];
ry(1.1051635894595386) q[1];
cx q[0],q[1];
ry(2.010800143265553) q[0];
ry(-0.17774533144160198) q[1];
cx q[0],q[1];
ry(1.3889576110210404) q[2];
ry(1.054261072988484) q[3];
cx q[2],q[3];
ry(-0.06179100610058758) q[2];
ry(1.8446950603068033) q[3];
cx q[2],q[3];
ry(-2.353833709486758) q[4];
ry(-2.4089993760683024) q[5];
cx q[4],q[5];
ry(0.9253378233590447) q[4];
ry(2.254095654003038) q[5];
cx q[4],q[5];
ry(-2.7020846250760693) q[6];
ry(-0.16252493917169453) q[7];
cx q[6],q[7];
ry(1.700923109575841) q[6];
ry(0.740942643174221) q[7];
cx q[6],q[7];
ry(-0.9773720355008049) q[0];
ry(-2.433531537355381) q[2];
cx q[0],q[2];
ry(2.538086462578266) q[0];
ry(-1.280398707222344) q[2];
cx q[0],q[2];
ry(-0.35792025828733376) q[2];
ry(-2.2212610770479237) q[4];
cx q[2],q[4];
ry(2.9504096265895474) q[2];
ry(1.6525783710233757) q[4];
cx q[2],q[4];
ry(-0.6903288140789579) q[4];
ry(1.1923922778220257) q[6];
cx q[4],q[6];
ry(2.6158842120308172) q[4];
ry(2.2753501833953784) q[6];
cx q[4],q[6];
ry(0.9247591566992233) q[1];
ry(-1.044272073719433) q[3];
cx q[1],q[3];
ry(0.09903962763005315) q[1];
ry(-1.4183668432677763) q[3];
cx q[1],q[3];
ry(1.454393299096588) q[3];
ry(-0.6194642705891678) q[5];
cx q[3],q[5];
ry(2.2212100348562274) q[3];
ry(2.1356787665062678) q[5];
cx q[3],q[5];
ry(-0.32948707099544183) q[5];
ry(-1.4258386729044723) q[7];
cx q[5],q[7];
ry(1.0703738653489672) q[5];
ry(-2.0741819658573184) q[7];
cx q[5],q[7];
ry(-2.0135831089366083) q[0];
ry(-2.7201499974728445) q[1];
cx q[0],q[1];
ry(2.6120762579417436) q[0];
ry(-3.1137179356595603) q[1];
cx q[0],q[1];
ry(2.5743231846455927) q[2];
ry(1.3161620132691103) q[3];
cx q[2],q[3];
ry(2.5039351285810967) q[2];
ry(0.5880161176158101) q[3];
cx q[2],q[3];
ry(-2.7946254298812585) q[4];
ry(1.0868556706132635) q[5];
cx q[4],q[5];
ry(-3.0542238298285804) q[4];
ry(1.9124322806206662) q[5];
cx q[4],q[5];
ry(-1.076279339728301) q[6];
ry(-1.4649753338716485) q[7];
cx q[6],q[7];
ry(-0.22297218004680575) q[6];
ry(2.3359812672935716) q[7];
cx q[6],q[7];
ry(1.7592823496284304) q[0];
ry(0.20650676060214862) q[2];
cx q[0],q[2];
ry(-0.4948270456586119) q[0];
ry(-2.245576656828554) q[2];
cx q[0],q[2];
ry(0.814494377630611) q[2];
ry(1.0996989256575982) q[4];
cx q[2],q[4];
ry(2.0171262703591535) q[2];
ry(1.8531767628005733) q[4];
cx q[2],q[4];
ry(-2.7385110779231034) q[4];
ry(2.0339484582372567) q[6];
cx q[4],q[6];
ry(0.2922515332695452) q[4];
ry(1.7032007065553714) q[6];
cx q[4],q[6];
ry(-1.371570755089837) q[1];
ry(0.4916469460953321) q[3];
cx q[1],q[3];
ry(2.77427367514304) q[1];
ry(2.1412309584711) q[3];
cx q[1],q[3];
ry(-1.9280545974494485) q[3];
ry(-2.8815161284614024) q[5];
cx q[3],q[5];
ry(0.7558214088991626) q[3];
ry(2.662167351894656) q[5];
cx q[3],q[5];
ry(2.067611497275587) q[5];
ry(-3.116505841688724) q[7];
cx q[5],q[7];
ry(-0.5963835782598439) q[5];
ry(-1.6503792672104467) q[7];
cx q[5],q[7];
ry(2.5177830488715274) q[0];
ry(-0.1080860062071138) q[1];
cx q[0],q[1];
ry(-1.501220982976542) q[0];
ry(-1.3268258863291325) q[1];
cx q[0],q[1];
ry(-0.03567912323085448) q[2];
ry(-1.1414576226703814) q[3];
cx q[2],q[3];
ry(1.8824502921393818) q[2];
ry(0.8476645863229901) q[3];
cx q[2],q[3];
ry(-3.002367194037986) q[4];
ry(1.3535208210896197) q[5];
cx q[4],q[5];
ry(0.14611714023725333) q[4];
ry(-0.27437469227752825) q[5];
cx q[4],q[5];
ry(-1.226653517482591) q[6];
ry(2.1654669095434453) q[7];
cx q[6],q[7];
ry(-2.706554196099291) q[6];
ry(-2.6253223556242866) q[7];
cx q[6],q[7];
ry(-0.1906861683401215) q[0];
ry(1.8455689116389553) q[2];
cx q[0],q[2];
ry(1.9238044848029903) q[0];
ry(0.4428180606817537) q[2];
cx q[0],q[2];
ry(-0.9428523159464886) q[2];
ry(0.9117134201132963) q[4];
cx q[2],q[4];
ry(-1.7622408090176251) q[2];
ry(-2.4250703124720965) q[4];
cx q[2],q[4];
ry(2.5968314242701696) q[4];
ry(-0.2364147761239688) q[6];
cx q[4],q[6];
ry(2.578983687932737) q[4];
ry(1.1901795308382468) q[6];
cx q[4],q[6];
ry(-3.026232847577862) q[1];
ry(-1.554651165034076) q[3];
cx q[1],q[3];
ry(-1.4110142655602849) q[1];
ry(-2.817452550796914) q[3];
cx q[1],q[3];
ry(0.12373097920516596) q[3];
ry(-0.657122734979299) q[5];
cx q[3],q[5];
ry(-0.4375645612496271) q[3];
ry(1.6112221845622632) q[5];
cx q[3],q[5];
ry(2.8754435575423427) q[5];
ry(-2.969341789702414) q[7];
cx q[5],q[7];
ry(0.6103437838741951) q[5];
ry(0.6599338443825683) q[7];
cx q[5],q[7];
ry(1.1689847332982284) q[0];
ry(-0.7974171228653978) q[1];
cx q[0],q[1];
ry(-3.0325836975732634) q[0];
ry(0.3937849300361602) q[1];
cx q[0],q[1];
ry(1.9212193696574964) q[2];
ry(-1.9154143914373039) q[3];
cx q[2],q[3];
ry(-0.9716007123501216) q[2];
ry(2.9783281977251983) q[3];
cx q[2],q[3];
ry(0.315467405596916) q[4];
ry(-2.3270925984148816) q[5];
cx q[4],q[5];
ry(-0.441774486150692) q[4];
ry(0.5654769235649253) q[5];
cx q[4],q[5];
ry(-2.488665647351575) q[6];
ry(2.833219784516705) q[7];
cx q[6],q[7];
ry(1.747260233403093) q[6];
ry(2.5559583246414843) q[7];
cx q[6],q[7];
ry(-0.618096842118903) q[0];
ry(-1.8933708013572739) q[2];
cx q[0],q[2];
ry(-2.490897284095813) q[0];
ry(-1.8306402785887235) q[2];
cx q[0],q[2];
ry(0.7305277244109809) q[2];
ry(-0.10362673372101427) q[4];
cx q[2],q[4];
ry(1.329150916791816) q[2];
ry(-2.300388383910422) q[4];
cx q[2],q[4];
ry(-1.4721419588926927) q[4];
ry(0.46662464741236054) q[6];
cx q[4],q[6];
ry(-1.7976334681598014) q[4];
ry(2.8574745298154975) q[6];
cx q[4],q[6];
ry(-2.649225827103496) q[1];
ry(-0.21381173285030639) q[3];
cx q[1],q[3];
ry(-0.19691731162974155) q[1];
ry(0.8433111684642549) q[3];
cx q[1],q[3];
ry(2.187063538321614) q[3];
ry(-0.6775431188162125) q[5];
cx q[3],q[5];
ry(2.108033795115419) q[3];
ry(2.326212188568317) q[5];
cx q[3],q[5];
ry(-1.7740646741416457) q[5];
ry(0.21756630886424322) q[7];
cx q[5],q[7];
ry(-2.6904565762373367) q[5];
ry(0.225913973738213) q[7];
cx q[5],q[7];
ry(1.3177229388026666) q[0];
ry(-1.7448120992741583) q[1];
cx q[0],q[1];
ry(-1.781077068302609) q[0];
ry(1.9544997865976625) q[1];
cx q[0],q[1];
ry(-0.8845496686193001) q[2];
ry(0.6870108856438861) q[3];
cx q[2],q[3];
ry(0.6654974512415757) q[2];
ry(1.1667366566843806) q[3];
cx q[2],q[3];
ry(0.9901969363246019) q[4];
ry(-3.1071179708479635) q[5];
cx q[4],q[5];
ry(1.1707977253604325) q[4];
ry(0.4028671120120247) q[5];
cx q[4],q[5];
ry(-1.9129550598570857) q[6];
ry(2.6336211951538515) q[7];
cx q[6],q[7];
ry(0.5532648096116324) q[6];
ry(2.728966402542312) q[7];
cx q[6],q[7];
ry(-0.22678829295662023) q[0];
ry(2.054563121130136) q[2];
cx q[0],q[2];
ry(-2.5603907482141506) q[0];
ry(1.9221659288429875) q[2];
cx q[0],q[2];
ry(0.2176695854507047) q[2];
ry(-2.3161487800777656) q[4];
cx q[2],q[4];
ry(-0.9802875137816702) q[2];
ry(-1.8500111034170077) q[4];
cx q[2],q[4];
ry(-2.829250145306257) q[4];
ry(0.40574539625464956) q[6];
cx q[4],q[6];
ry(-1.4209644238812587) q[4];
ry(-1.9003704528968548) q[6];
cx q[4],q[6];
ry(-1.2640893101976298) q[1];
ry(-1.7848177422379694) q[3];
cx q[1],q[3];
ry(-0.06623665812190466) q[1];
ry(1.2215435509507158) q[3];
cx q[1],q[3];
ry(0.4602656614485643) q[3];
ry(1.6736429678893225) q[5];
cx q[3],q[5];
ry(-3.084809098962658) q[3];
ry(-0.5498311548346679) q[5];
cx q[3],q[5];
ry(0.7263580064360814) q[5];
ry(-2.325694276653587) q[7];
cx q[5],q[7];
ry(2.4078764964078103) q[5];
ry(-2.0155707985140183) q[7];
cx q[5],q[7];
ry(3.076830749211856) q[0];
ry(1.3168731651087466) q[1];
cx q[0],q[1];
ry(-0.40864523160978106) q[0];
ry(-2.9105102470850888) q[1];
cx q[0],q[1];
ry(1.1068595110663297) q[2];
ry(2.852412592423135) q[3];
cx q[2],q[3];
ry(-1.4834290396369976) q[2];
ry(-0.9871614434114484) q[3];
cx q[2],q[3];
ry(3.0894424964630023) q[4];
ry(2.525614818082999) q[5];
cx q[4],q[5];
ry(1.5289569202269693) q[4];
ry(-3.0742528978459105) q[5];
cx q[4],q[5];
ry(-1.5089624849018475) q[6];
ry(0.6456083463528264) q[7];
cx q[6],q[7];
ry(2.5055897819696766) q[6];
ry(2.4595628776726115) q[7];
cx q[6],q[7];
ry(0.9437408655663324) q[0];
ry(2.871634433065509) q[2];
cx q[0],q[2];
ry(1.4739958692519668) q[0];
ry(2.585888251343828) q[2];
cx q[0],q[2];
ry(-1.8406128252961833) q[2];
ry(-2.706360645422185) q[4];
cx q[2],q[4];
ry(2.678585993754952) q[2];
ry(-2.8388588489415367) q[4];
cx q[2],q[4];
ry(1.8572403027194178) q[4];
ry(-1.7993555730413997) q[6];
cx q[4],q[6];
ry(-2.515168975955093) q[4];
ry(-2.4351438027670675) q[6];
cx q[4],q[6];
ry(0.17031255603427026) q[1];
ry(-0.04016465911523337) q[3];
cx q[1],q[3];
ry(-0.41435148757992396) q[1];
ry(-1.344661231339098) q[3];
cx q[1],q[3];
ry(0.7591494122812348) q[3];
ry(0.7274700600239159) q[5];
cx q[3],q[5];
ry(1.5597138624395843) q[3];
ry(-0.33616300452732956) q[5];
cx q[3],q[5];
ry(0.3621566988573921) q[5];
ry(-0.2438187898473556) q[7];
cx q[5],q[7];
ry(2.760958180393594) q[5];
ry(-1.3010555222890432) q[7];
cx q[5],q[7];
ry(1.1353505012369451) q[0];
ry(-0.3376248736507428) q[1];
cx q[0],q[1];
ry(0.1699489208552647) q[0];
ry(-0.6891066344655146) q[1];
cx q[0],q[1];
ry(2.7589433785652986) q[2];
ry(3.0264520059537054) q[3];
cx q[2],q[3];
ry(-2.5697709001340123) q[2];
ry(1.494658562869599) q[3];
cx q[2],q[3];
ry(-1.8120558676133196) q[4];
ry(-1.7444484116833456) q[5];
cx q[4],q[5];
ry(2.2338993894116834) q[4];
ry(0.17552658113783615) q[5];
cx q[4],q[5];
ry(1.3580125667559104) q[6];
ry(-3.132689188538928) q[7];
cx q[6],q[7];
ry(-2.7669921480366284) q[6];
ry(2.5346598422629683) q[7];
cx q[6],q[7];
ry(-1.4192580996708792) q[0];
ry(-0.6782531170675083) q[2];
cx q[0],q[2];
ry(1.383558709998264) q[0];
ry(-3.0366801660482388) q[2];
cx q[0],q[2];
ry(-0.2967706631261703) q[2];
ry(0.5724679684551814) q[4];
cx q[2],q[4];
ry(-2.131464919913576) q[2];
ry(1.7331215196120278) q[4];
cx q[2],q[4];
ry(-1.453119132313426) q[4];
ry(-2.170704822625866) q[6];
cx q[4],q[6];
ry(-2.432594341691362) q[4];
ry(-2.445541743667313) q[6];
cx q[4],q[6];
ry(0.6112052038556909) q[1];
ry(-2.782007667549402) q[3];
cx q[1],q[3];
ry(-0.8566075333731761) q[1];
ry(-1.0686990883230008) q[3];
cx q[1],q[3];
ry(2.9033224937028845) q[3];
ry(2.120686970783355) q[5];
cx q[3],q[5];
ry(2.1089754375834335) q[3];
ry(1.0277926065359333) q[5];
cx q[3],q[5];
ry(1.8728734974858492) q[5];
ry(3.022671319022767) q[7];
cx q[5],q[7];
ry(-2.1859913792346273) q[5];
ry(2.809568898735024) q[7];
cx q[5],q[7];
ry(-1.6890768371106606) q[0];
ry(-2.164230365289371) q[1];
cx q[0],q[1];
ry(-1.4533319051917817) q[0];
ry(-2.638765367886592) q[1];
cx q[0],q[1];
ry(1.2926322320088843) q[2];
ry(0.9358697152884892) q[3];
cx q[2],q[3];
ry(-2.641357477118199) q[2];
ry(1.1712135242314057) q[3];
cx q[2],q[3];
ry(-2.66222667046426) q[4];
ry(0.3800974421954458) q[5];
cx q[4],q[5];
ry(0.5954305849142445) q[4];
ry(-2.262967429469791) q[5];
cx q[4],q[5];
ry(-3.1004054200783213) q[6];
ry(1.1981264645628142) q[7];
cx q[6],q[7];
ry(-0.8741696755599192) q[6];
ry(-2.3664430147930036) q[7];
cx q[6],q[7];
ry(1.7290455994364837) q[0];
ry(-2.8401799712418643) q[2];
cx q[0],q[2];
ry(-0.2833680671279936) q[0];
ry(1.4235966928616026) q[2];
cx q[0],q[2];
ry(-1.9765908683566966) q[2];
ry(-1.299155948123445) q[4];
cx q[2],q[4];
ry(-0.9459731443403561) q[2];
ry(-2.064382659890976) q[4];
cx q[2],q[4];
ry(-1.3000205379172778) q[4];
ry(2.039701256818748) q[6];
cx q[4],q[6];
ry(-0.8566716024026941) q[4];
ry(-1.808212426142292) q[6];
cx q[4],q[6];
ry(-2.042283900941417) q[1];
ry(-1.8962798566634487) q[3];
cx q[1],q[3];
ry(2.1283404540083413) q[1];
ry(0.6707728450355832) q[3];
cx q[1],q[3];
ry(-0.3420883651392406) q[3];
ry(0.9492362650300405) q[5];
cx q[3],q[5];
ry(-2.9368763620967187) q[3];
ry(-0.3009915640090508) q[5];
cx q[3],q[5];
ry(-2.941759098086347) q[5];
ry(-1.8697769549549077) q[7];
cx q[5],q[7];
ry(1.7674673135905208) q[5];
ry(-1.011032018637037) q[7];
cx q[5],q[7];
ry(-0.5359413849846497) q[0];
ry(0.44238398636317366) q[1];
cx q[0],q[1];
ry(1.2282332799221742) q[0];
ry(0.740589840004583) q[1];
cx q[0],q[1];
ry(-2.5313395901182174) q[2];
ry(-1.152812493623271) q[3];
cx q[2],q[3];
ry(1.5072742924512246) q[2];
ry(0.3933357483685995) q[3];
cx q[2],q[3];
ry(-0.03747411619991414) q[4];
ry(3.1087735142191866) q[5];
cx q[4],q[5];
ry(-1.7852057778933783) q[4];
ry(1.4506763203504571) q[5];
cx q[4],q[5];
ry(1.7115714436454743) q[6];
ry(2.1335771829566927) q[7];
cx q[6],q[7];
ry(2.0002432432524038) q[6];
ry(0.5233145722881722) q[7];
cx q[6],q[7];
ry(1.3452786688506835) q[0];
ry(1.8514414698243604) q[2];
cx q[0],q[2];
ry(1.199601370441333) q[0];
ry(-2.33365864001353) q[2];
cx q[0],q[2];
ry(2.855041074775362) q[2];
ry(1.886112425044429) q[4];
cx q[2],q[4];
ry(0.031185577005069284) q[2];
ry(2.4976707204918407) q[4];
cx q[2],q[4];
ry(-0.9789708735906465) q[4];
ry(3.0447149077745994) q[6];
cx q[4],q[6];
ry(2.6081910529801435) q[4];
ry(2.637733654862576) q[6];
cx q[4],q[6];
ry(-1.7967876898064479) q[1];
ry(1.27572617405281) q[3];
cx q[1],q[3];
ry(-2.327649967542103) q[1];
ry(2.7968255770213677) q[3];
cx q[1],q[3];
ry(-2.3588157073271567) q[3];
ry(-0.6173785690004223) q[5];
cx q[3],q[5];
ry(-2.654399665148955) q[3];
ry(-2.87156784184727) q[5];
cx q[3],q[5];
ry(0.5896964123150045) q[5];
ry(-1.579582662856228) q[7];
cx q[5],q[7];
ry(-2.2726958009893714) q[5];
ry(2.7980601753780485) q[7];
cx q[5],q[7];
ry(2.2807759512160857) q[0];
ry(2.872564492139822) q[1];
cx q[0],q[1];
ry(-3.056238279815829) q[0];
ry(2.207769968004782) q[1];
cx q[0],q[1];
ry(-1.6651824825675707) q[2];
ry(1.3058791397696847) q[3];
cx q[2],q[3];
ry(2.6174163023867068) q[2];
ry(-0.9190085101025387) q[3];
cx q[2],q[3];
ry(2.2854186377595536) q[4];
ry(-0.5684042880728957) q[5];
cx q[4],q[5];
ry(-2.656629289712855) q[4];
ry(2.9958130223355095) q[5];
cx q[4],q[5];
ry(-1.0325933891298789) q[6];
ry(-0.6925827646922705) q[7];
cx q[6],q[7];
ry(-2.970437200269027) q[6];
ry(2.3133744220517687) q[7];
cx q[6],q[7];
ry(0.012692801937430723) q[0];
ry(-1.6192377650811025) q[2];
cx q[0],q[2];
ry(-0.6744047702824378) q[0];
ry(-2.7379529978876893) q[2];
cx q[0],q[2];
ry(-0.8742432965705219) q[2];
ry(0.5242045508862598) q[4];
cx q[2],q[4];
ry(0.9117370306698831) q[2];
ry(-2.0145507970606715) q[4];
cx q[2],q[4];
ry(-0.7529480049362405) q[4];
ry(1.1945188127764759) q[6];
cx q[4],q[6];
ry(-0.4237787864639966) q[4];
ry(1.618657944986622) q[6];
cx q[4],q[6];
ry(-3.095221852538421) q[1];
ry(3.0798723261999323) q[3];
cx q[1],q[3];
ry(-2.4870582381449187) q[1];
ry(-3.134229655558337) q[3];
cx q[1],q[3];
ry(0.5757697859721659) q[3];
ry(-0.6425420521454432) q[5];
cx q[3],q[5];
ry(1.2590240946846167) q[3];
ry(-0.7006175729210886) q[5];
cx q[3],q[5];
ry(1.2871286156718333) q[5];
ry(-1.3987616623321992) q[7];
cx q[5],q[7];
ry(-1.401411136992431) q[5];
ry(-3.0681565668719886) q[7];
cx q[5],q[7];
ry(2.803372425099924) q[0];
ry(2.678608669261452) q[1];
cx q[0],q[1];
ry(2.9263406454036374) q[0];
ry(0.1462008337231507) q[1];
cx q[0],q[1];
ry(2.9030992687213892) q[2];
ry(-2.9340770501740128) q[3];
cx q[2],q[3];
ry(1.4083372446426863) q[2];
ry(2.0551194677682485) q[3];
cx q[2],q[3];
ry(-1.8513762571776045) q[4];
ry(0.6352181670181593) q[5];
cx q[4],q[5];
ry(1.1947415957115834) q[4];
ry(-0.7178897483891797) q[5];
cx q[4],q[5];
ry(-0.31616826790353825) q[6];
ry(-0.4376973809625336) q[7];
cx q[6],q[7];
ry(3.0345832597619853) q[6];
ry(-2.960422469338266) q[7];
cx q[6],q[7];
ry(1.757033626314667) q[0];
ry(0.37508316983967754) q[2];
cx q[0],q[2];
ry(-1.4567974090999511) q[0];
ry(1.99610329633373) q[2];
cx q[0],q[2];
ry(2.4521047033520795) q[2];
ry(-1.0485911941502193) q[4];
cx q[2],q[4];
ry(0.4848054098754888) q[2];
ry(-0.012246450520570384) q[4];
cx q[2],q[4];
ry(0.03667083882611788) q[4];
ry(1.0222143680392) q[6];
cx q[4],q[6];
ry(1.6775529675512866) q[4];
ry(-0.9667254575166431) q[6];
cx q[4],q[6];
ry(1.62794924078124) q[1];
ry(-2.912776035372631) q[3];
cx q[1],q[3];
ry(-2.318760860332663) q[1];
ry(-0.38437763548305676) q[3];
cx q[1],q[3];
ry(-2.0916506154867367) q[3];
ry(2.2632773744545145) q[5];
cx q[3],q[5];
ry(-2.5113882327326578) q[3];
ry(-2.432125047099755) q[5];
cx q[3],q[5];
ry(2.766513254861528) q[5];
ry(-2.5839732326740346) q[7];
cx q[5],q[7];
ry(-2.7687864329196645) q[5];
ry(0.05489764964244248) q[7];
cx q[5],q[7];
ry(0.9096499853087499) q[0];
ry(0.030317971123961836) q[1];
cx q[0],q[1];
ry(-2.5099117624157703) q[0];
ry(1.4705825047370211) q[1];
cx q[0],q[1];
ry(2.5213791573030075) q[2];
ry(-2.005352537434502) q[3];
cx q[2],q[3];
ry(-2.8778061265026462) q[2];
ry(-0.6906043894760557) q[3];
cx q[2],q[3];
ry(-2.1810939698447536) q[4];
ry(2.8054084353207167) q[5];
cx q[4],q[5];
ry(1.7774470659266255) q[4];
ry(-1.0449510681642218) q[5];
cx q[4],q[5];
ry(-0.4704226698592837) q[6];
ry(-2.96131736955276) q[7];
cx q[6],q[7];
ry(0.9303208748471248) q[6];
ry(-0.7259328708220415) q[7];
cx q[6],q[7];
ry(-3.0289760453836) q[0];
ry(-1.7261071419648322) q[2];
cx q[0],q[2];
ry(-1.7618525340342828) q[0];
ry(-1.7007232439794784) q[2];
cx q[0],q[2];
ry(-0.8750930273685293) q[2];
ry(-0.6335102200475524) q[4];
cx q[2],q[4];
ry(1.4965547279118732) q[2];
ry(-0.9027962668479454) q[4];
cx q[2],q[4];
ry(-1.7432305142742777) q[4];
ry(2.403285961947116) q[6];
cx q[4],q[6];
ry(-2.40050906132794) q[4];
ry(2.912482624374017) q[6];
cx q[4],q[6];
ry(0.7715562925157575) q[1];
ry(-1.3387574179539836) q[3];
cx q[1],q[3];
ry(-0.5363280177190344) q[1];
ry(1.622518648624275) q[3];
cx q[1],q[3];
ry(-2.202460864061945) q[3];
ry(-0.9280035909559978) q[5];
cx q[3],q[5];
ry(-1.7038678622681749) q[3];
ry(-1.5781287581270993) q[5];
cx q[3],q[5];
ry(-0.3576765416141905) q[5];
ry(2.91761875111881) q[7];
cx q[5],q[7];
ry(2.6666954976268467) q[5];
ry(-0.5006431746783342) q[7];
cx q[5],q[7];
ry(-2.492147805436258) q[0];
ry(-1.803710445656285) q[1];
cx q[0],q[1];
ry(3.058520615461293) q[0];
ry(-2.274486236277602) q[1];
cx q[0],q[1];
ry(1.668074244331794) q[2];
ry(-2.1096371868713835) q[3];
cx q[2],q[3];
ry(2.828709729786472) q[2];
ry(3.1290897211384556) q[3];
cx q[2],q[3];
ry(-0.4357360029776828) q[4];
ry(-1.3590228544819853) q[5];
cx q[4],q[5];
ry(-2.945368810022526) q[4];
ry(1.6131041155725616) q[5];
cx q[4],q[5];
ry(1.6010743432904562) q[6];
ry(-0.4236224555742485) q[7];
cx q[6],q[7];
ry(1.9175901201550456) q[6];
ry(-1.2310327728073895) q[7];
cx q[6],q[7];
ry(-0.3371292038174763) q[0];
ry(0.5875979992783081) q[2];
cx q[0],q[2];
ry(0.19406555544707338) q[0];
ry(1.5393431952431258) q[2];
cx q[0],q[2];
ry(0.6136315508234285) q[2];
ry(0.3802709520988929) q[4];
cx q[2],q[4];
ry(1.2402713742715148) q[2];
ry(-0.98377812997622) q[4];
cx q[2],q[4];
ry(-1.1360114097395684) q[4];
ry(-0.37512090634045886) q[6];
cx q[4],q[6];
ry(-0.2591242029791996) q[4];
ry(1.932868582549311) q[6];
cx q[4],q[6];
ry(1.9237752640303176) q[1];
ry(1.046585350117215) q[3];
cx q[1],q[3];
ry(-2.1224593285818765) q[1];
ry(-2.342323474952066) q[3];
cx q[1],q[3];
ry(0.5806151062032054) q[3];
ry(0.937690362238218) q[5];
cx q[3],q[5];
ry(-1.6714700394749782) q[3];
ry(-2.2495716990112005) q[5];
cx q[3],q[5];
ry(1.7818347762742963) q[5];
ry(-1.0375082603202834) q[7];
cx q[5],q[7];
ry(0.057002895147908746) q[5];
ry(0.4245745510073739) q[7];
cx q[5],q[7];
ry(-0.4427970582506081) q[0];
ry(1.7953032229701358) q[1];
cx q[0],q[1];
ry(0.9929848409931692) q[0];
ry(-2.9500014729656274) q[1];
cx q[0],q[1];
ry(-2.990409056884323) q[2];
ry(0.6390650901009608) q[3];
cx q[2],q[3];
ry(3.077320678084366) q[2];
ry(-1.484569244181685) q[3];
cx q[2],q[3];
ry(2.457778229854089) q[4];
ry(1.0499246996960148) q[5];
cx q[4],q[5];
ry(-0.2847051058010484) q[4];
ry(1.879450820742562) q[5];
cx q[4],q[5];
ry(3.1113737432071504) q[6];
ry(0.3897692971221449) q[7];
cx q[6],q[7];
ry(-1.8954961683821638) q[6];
ry(-1.9646465003536013) q[7];
cx q[6],q[7];
ry(-0.6040563256650274) q[0];
ry(1.366842586014552) q[2];
cx q[0],q[2];
ry(-2.636018565493398) q[0];
ry(0.889914523821699) q[2];
cx q[0],q[2];
ry(-1.0213115858975887) q[2];
ry(-1.2708284151497198) q[4];
cx q[2],q[4];
ry(3.0592488727527627) q[2];
ry(0.4762713998437347) q[4];
cx q[2],q[4];
ry(2.2784876120838073) q[4];
ry(-1.6216204001110501) q[6];
cx q[4],q[6];
ry(-1.495677592630764) q[4];
ry(-0.4425144654482853) q[6];
cx q[4],q[6];
ry(-1.4660329891240966) q[1];
ry(1.5522046516559629) q[3];
cx q[1],q[3];
ry(2.004007811047467) q[1];
ry(0.6987682943602592) q[3];
cx q[1],q[3];
ry(-0.7034837933237267) q[3];
ry(-1.2531988217729948) q[5];
cx q[3],q[5];
ry(0.5014467839440815) q[3];
ry(2.76156574440363) q[5];
cx q[3],q[5];
ry(2.1816203502599416) q[5];
ry(-0.512182290457833) q[7];
cx q[5],q[7];
ry(1.9080637636073112) q[5];
ry(2.3471314317993985) q[7];
cx q[5],q[7];
ry(-1.7784192845988036) q[0];
ry(0.24805695823548313) q[1];
cx q[0],q[1];
ry(-0.1394442542450551) q[0];
ry(0.3988340108518127) q[1];
cx q[0],q[1];
ry(-0.8954873322205639) q[2];
ry(-0.19757880367976302) q[3];
cx q[2],q[3];
ry(-1.1574137654276342) q[2];
ry(-1.38960125657052) q[3];
cx q[2],q[3];
ry(-2.1434415536362637) q[4];
ry(-2.568668914863229) q[5];
cx q[4],q[5];
ry(0.25903748281840766) q[4];
ry(0.23076171002720347) q[5];
cx q[4],q[5];
ry(-0.5828522215331144) q[6];
ry(2.325937088190551) q[7];
cx q[6],q[7];
ry(2.8000833729031775) q[6];
ry(-1.4606297261956884) q[7];
cx q[6],q[7];
ry(0.7436281920221253) q[0];
ry(-0.8626150513635931) q[2];
cx q[0],q[2];
ry(2.5958672704297086) q[0];
ry(1.2586434936634632) q[2];
cx q[0],q[2];
ry(-1.7167537561660513) q[2];
ry(-2.2086390620359113) q[4];
cx q[2],q[4];
ry(2.5957935988893768) q[2];
ry(2.618622627483521) q[4];
cx q[2],q[4];
ry(-2.6907502339106344) q[4];
ry(2.7686354973540306) q[6];
cx q[4],q[6];
ry(3.1135836081688137) q[4];
ry(0.8810312741708026) q[6];
cx q[4],q[6];
ry(1.9213219880647143) q[1];
ry(-0.25377561704787543) q[3];
cx q[1],q[3];
ry(1.879249289977884) q[1];
ry(0.25755344613798814) q[3];
cx q[1],q[3];
ry(-2.0750963693063373) q[3];
ry(2.4077814280819387) q[5];
cx q[3],q[5];
ry(-1.6731251279604449) q[3];
ry(1.350514410430418) q[5];
cx q[3],q[5];
ry(-1.6118083431322594) q[5];
ry(0.4855502026831104) q[7];
cx q[5],q[7];
ry(0.09524025012148485) q[5];
ry(-2.205335230099385) q[7];
cx q[5],q[7];
ry(-2.45751763163023) q[0];
ry(-2.602165431947027) q[1];
ry(-1.907207277026762) q[2];
ry(0.5934286959813733) q[3];
ry(2.6872311428269757) q[4];
ry(-1.220129698833188) q[5];
ry(1.8000707758594594) q[6];
ry(-2.5143904922501763) q[7];