OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(2.8019228401212333) q[0];
ry(2.590866407316338) q[1];
cx q[0],q[1];
ry(1.2745218895051953) q[0];
ry(2.847738713547304) q[1];
cx q[0],q[1];
ry(1.8834250795231364) q[0];
ry(2.079146577069353) q[2];
cx q[0],q[2];
ry(0.0402008837995691) q[0];
ry(-3.063983946657854) q[2];
cx q[0],q[2];
ry(-1.4151803406292585) q[0];
ry(2.995852943431387) q[3];
cx q[0],q[3];
ry(-1.6097511541229927) q[0];
ry(0.1273617539661254) q[3];
cx q[0],q[3];
ry(-1.5415349730159773) q[0];
ry(1.860017710184815) q[4];
cx q[0],q[4];
ry(-0.3831532999295942) q[0];
ry(0.9086540122114379) q[4];
cx q[0],q[4];
ry(2.531765191366475) q[0];
ry(0.09813991934471833) q[5];
cx q[0],q[5];
ry(-2.3242352609865335) q[0];
ry(-1.5658333728338727) q[5];
cx q[0],q[5];
ry(-2.3435448083207553) q[0];
ry(-0.3026273791853997) q[6];
cx q[0],q[6];
ry(2.0524736733675595) q[0];
ry(-1.7682805771260217) q[6];
cx q[0],q[6];
ry(-2.0080876468932436) q[0];
ry(-0.39193015393445574) q[7];
cx q[0],q[7];
ry(-3.0865973949034813) q[0];
ry(-1.2307774434620855) q[7];
cx q[0],q[7];
ry(-1.8524851876482211) q[1];
ry(-0.8495321855110815) q[2];
cx q[1],q[2];
ry(3.0731362072143114) q[1];
ry(-2.3309331866103857) q[2];
cx q[1],q[2];
ry(1.8441174746863347) q[1];
ry(-0.49074183318815856) q[3];
cx q[1],q[3];
ry(1.3032772091486668) q[1];
ry(1.865923664935746) q[3];
cx q[1],q[3];
ry(0.3456993653698322) q[1];
ry(1.0247878667372445) q[4];
cx q[1],q[4];
ry(2.708921805489669) q[1];
ry(-0.5596401203215882) q[4];
cx q[1],q[4];
ry(0.004911964032213945) q[1];
ry(-2.345716016191893) q[5];
cx q[1],q[5];
ry(-1.4454004860746397) q[1];
ry(-0.6110483851268276) q[5];
cx q[1],q[5];
ry(2.529705587138482) q[1];
ry(-0.5160945025885866) q[6];
cx q[1],q[6];
ry(2.920663729839475) q[1];
ry(0.655862258194583) q[6];
cx q[1],q[6];
ry(2.422667554341413) q[1];
ry(0.026667911110177304) q[7];
cx q[1],q[7];
ry(1.344159649664522) q[1];
ry(-1.073991056374536) q[7];
cx q[1],q[7];
ry(-2.4329035909103673) q[2];
ry(-3.010826637304432) q[3];
cx q[2],q[3];
ry(-0.1696542386011748) q[2];
ry(3.051674408477617) q[3];
cx q[2],q[3];
ry(0.9221951654595559) q[2];
ry(-0.5978860987449427) q[4];
cx q[2],q[4];
ry(-1.7741443729826278) q[2];
ry(0.536663231720329) q[4];
cx q[2],q[4];
ry(1.2091731306067932) q[2];
ry(-2.447468108197952) q[5];
cx q[2],q[5];
ry(-0.3429472630151181) q[2];
ry(2.54789808054459) q[5];
cx q[2],q[5];
ry(0.8192322046137108) q[2];
ry(-2.9154383224430687) q[6];
cx q[2],q[6];
ry(0.22736699870425348) q[2];
ry(-2.9953235313955595) q[6];
cx q[2],q[6];
ry(-0.21459246661970002) q[2];
ry(-0.19642040703878383) q[7];
cx q[2],q[7];
ry(-1.574449866097682) q[2];
ry(1.066162487174793) q[7];
cx q[2],q[7];
ry(-1.6170616113466574) q[3];
ry(-2.0375967563810122) q[4];
cx q[3],q[4];
ry(-2.765979149370195) q[3];
ry(-0.6921174805789139) q[4];
cx q[3],q[4];
ry(1.6263234412281733) q[3];
ry(2.5158028018947594) q[5];
cx q[3],q[5];
ry(-2.386186598512191) q[3];
ry(-0.07082852119788051) q[5];
cx q[3],q[5];
ry(-2.110454532067234) q[3];
ry(-0.3202369141972525) q[6];
cx q[3],q[6];
ry(-1.4846325003546479) q[3];
ry(1.1242230279095977) q[6];
cx q[3],q[6];
ry(2.207467794190802) q[3];
ry(2.8979022617539814) q[7];
cx q[3],q[7];
ry(0.8211163381437062) q[3];
ry(-1.701923612049712) q[7];
cx q[3],q[7];
ry(-2.051196875389242) q[4];
ry(-2.8373369189156366) q[5];
cx q[4],q[5];
ry(0.29506368797737714) q[4];
ry(-1.6010499865073342) q[5];
cx q[4],q[5];
ry(-2.363616171766168) q[4];
ry(1.0753576195249857) q[6];
cx q[4],q[6];
ry(-1.1255817283406675) q[4];
ry(0.9736987322880454) q[6];
cx q[4],q[6];
ry(1.1090142238589156) q[4];
ry(0.0025684793140650086) q[7];
cx q[4],q[7];
ry(-1.2369606689336414) q[4];
ry(-0.823397195795612) q[7];
cx q[4],q[7];
ry(-0.5602824215814443) q[5];
ry(0.7204483587799555) q[6];
cx q[5],q[6];
ry(1.379158215407787) q[5];
ry(1.2624324986308952) q[6];
cx q[5],q[6];
ry(0.15272090898298885) q[5];
ry(-1.414554764425458) q[7];
cx q[5],q[7];
ry(-2.9960950939874778) q[5];
ry(1.486877194266266) q[7];
cx q[5],q[7];
ry(0.9744284963286151) q[6];
ry(1.9999793728428426) q[7];
cx q[6],q[7];
ry(-2.789452170772246) q[6];
ry(2.7847252319518354) q[7];
cx q[6],q[7];
ry(-1.5252925480613186) q[0];
ry(-1.1188055064746156) q[1];
cx q[0],q[1];
ry(2.008194481887487) q[0];
ry(2.3359540912220567) q[1];
cx q[0],q[1];
ry(-0.0036877869866809247) q[0];
ry(1.2815695996056116) q[2];
cx q[0],q[2];
ry(1.398243145346136) q[0];
ry(0.11662435064374677) q[2];
cx q[0],q[2];
ry(-0.7468764926436443) q[0];
ry(-2.5751006574175666) q[3];
cx q[0],q[3];
ry(2.210249466113022) q[0];
ry(2.3570378837559383) q[3];
cx q[0],q[3];
ry(1.3180274599639537) q[0];
ry(3.1225695534370352) q[4];
cx q[0],q[4];
ry(-2.8296851563747873) q[0];
ry(-3.073607399236308) q[4];
cx q[0],q[4];
ry(1.5940105675010796) q[0];
ry(3.0484493803826043) q[5];
cx q[0],q[5];
ry(-2.632627020539128) q[0];
ry(2.8500194977509636) q[5];
cx q[0],q[5];
ry(-0.728185573129613) q[0];
ry(-1.0937414053175933) q[6];
cx q[0],q[6];
ry(-0.5519818393478171) q[0];
ry(-2.7429143406358194) q[6];
cx q[0],q[6];
ry(-1.492492515969614) q[0];
ry(-2.6706275166659066) q[7];
cx q[0],q[7];
ry(2.8674478785751405) q[0];
ry(-2.023791624400089) q[7];
cx q[0],q[7];
ry(-1.7530677316841206) q[1];
ry(-3.0342773961333) q[2];
cx q[1],q[2];
ry(-0.21959644399531886) q[1];
ry(-1.3693753901719505) q[2];
cx q[1],q[2];
ry(-1.353274247056568) q[1];
ry(-1.6030601533124387) q[3];
cx q[1],q[3];
ry(2.0330652132426055) q[1];
ry(2.8092125898997193) q[3];
cx q[1],q[3];
ry(-2.60305539681539) q[1];
ry(-0.22861446338803598) q[4];
cx q[1],q[4];
ry(0.9545831817035877) q[1];
ry(-2.7776686735142873) q[4];
cx q[1],q[4];
ry(-2.840534635024741) q[1];
ry(1.409518954635132) q[5];
cx q[1],q[5];
ry(2.840180821846442) q[1];
ry(1.7433439967958595) q[5];
cx q[1],q[5];
ry(-1.2930392882255495) q[1];
ry(-0.6035639519090088) q[6];
cx q[1],q[6];
ry(1.191246772088323) q[1];
ry(0.538091385221509) q[6];
cx q[1],q[6];
ry(-0.430066820869833) q[1];
ry(-2.6681199075176534) q[7];
cx q[1],q[7];
ry(-1.2860870384516483) q[1];
ry(-0.4685720701618781) q[7];
cx q[1],q[7];
ry(0.8941346380800601) q[2];
ry(-2.5164737377661908) q[3];
cx q[2],q[3];
ry(0.3262471076849388) q[2];
ry(3.065719337854792) q[3];
cx q[2],q[3];
ry(0.004291544182870588) q[2];
ry(2.7000705858488745) q[4];
cx q[2],q[4];
ry(-0.753016676317492) q[2];
ry(-2.641561767341764) q[4];
cx q[2],q[4];
ry(-0.36974060618328375) q[2];
ry(-1.9862394592080175) q[5];
cx q[2],q[5];
ry(-0.9863543544050681) q[2];
ry(-2.7837124858164546) q[5];
cx q[2],q[5];
ry(1.0026580045287166) q[2];
ry(-3.039168471926805) q[6];
cx q[2],q[6];
ry(0.7449188744268049) q[2];
ry(-0.2912292023154341) q[6];
cx q[2],q[6];
ry(0.9772285089901969) q[2];
ry(-2.6316958383767877) q[7];
cx q[2],q[7];
ry(1.4168083901590514) q[2];
ry(-1.9852379348880236) q[7];
cx q[2],q[7];
ry(-0.8085896932366534) q[3];
ry(-1.327963222212138) q[4];
cx q[3],q[4];
ry(-3.097678800219651) q[3];
ry(-1.3654237690188031) q[4];
cx q[3],q[4];
ry(0.7191885612572155) q[3];
ry(1.5125895124658097) q[5];
cx q[3],q[5];
ry(1.1957874323255289) q[3];
ry(-1.835487949029865) q[5];
cx q[3],q[5];
ry(2.8231527783464982) q[3];
ry(2.4401552518508858) q[6];
cx q[3],q[6];
ry(-1.0478213967469578) q[3];
ry(-2.562600572661811) q[6];
cx q[3],q[6];
ry(-1.2984284722436483) q[3];
ry(-1.0348155211199037) q[7];
cx q[3],q[7];
ry(-0.2960363766430149) q[3];
ry(1.0037341430642914) q[7];
cx q[3],q[7];
ry(-1.2409736016984967) q[4];
ry(0.9707176515440122) q[5];
cx q[4],q[5];
ry(-0.37973500998815535) q[4];
ry(3.1237413133314025) q[5];
cx q[4],q[5];
ry(3.010886395166028) q[4];
ry(-2.9787773635354804) q[6];
cx q[4],q[6];
ry(-0.5324263682295999) q[4];
ry(1.2408097729166823) q[6];
cx q[4],q[6];
ry(-2.675502893241498) q[4];
ry(-1.8457078105090918) q[7];
cx q[4],q[7];
ry(2.985210990610028) q[4];
ry(0.12985792421963185) q[7];
cx q[4],q[7];
ry(-1.019487081356968) q[5];
ry(0.24114962063112524) q[6];
cx q[5],q[6];
ry(-2.8746485849429493) q[5];
ry(-2.323858919402533) q[6];
cx q[5],q[6];
ry(-3.0989337356524906) q[5];
ry(0.6540906361458341) q[7];
cx q[5],q[7];
ry(-2.89937183596096) q[5];
ry(3.1139414037525817) q[7];
cx q[5],q[7];
ry(-1.7298308432645815) q[6];
ry(-1.944109987757729) q[7];
cx q[6],q[7];
ry(1.1759790485543644) q[6];
ry(-2.015223180278408) q[7];
cx q[6],q[7];
ry(-1.4111989935393643) q[0];
ry(1.4777093882101129) q[1];
cx q[0],q[1];
ry(2.7628933401940445) q[0];
ry(0.038400317071729084) q[1];
cx q[0],q[1];
ry(2.082721444359181) q[0];
ry(0.21466943712436334) q[2];
cx q[0],q[2];
ry(-0.08563748180048236) q[0];
ry(-1.424222916056178) q[2];
cx q[0],q[2];
ry(2.5677520801343143) q[0];
ry(2.382170405278383) q[3];
cx q[0],q[3];
ry(-0.3065332315253819) q[0];
ry(2.7256840524056383) q[3];
cx q[0],q[3];
ry(-0.6107606493981202) q[0];
ry(0.247733997100676) q[4];
cx q[0],q[4];
ry(2.338568604475239) q[0];
ry(-0.23401716497299763) q[4];
cx q[0],q[4];
ry(-0.20442076104794804) q[0];
ry(2.6009752561064374) q[5];
cx q[0],q[5];
ry(2.8026790614899295) q[0];
ry(2.3867646131803384) q[5];
cx q[0],q[5];
ry(-2.60016793729316) q[0];
ry(-2.3090871415714753) q[6];
cx q[0],q[6];
ry(1.598592483559216) q[0];
ry(0.07064494051275655) q[6];
cx q[0],q[6];
ry(-0.2526983828407712) q[0];
ry(-0.5786841508765589) q[7];
cx q[0],q[7];
ry(-0.4041628799357282) q[0];
ry(0.489047602389098) q[7];
cx q[0],q[7];
ry(2.0812661566542285) q[1];
ry(-2.367050367361146) q[2];
cx q[1],q[2];
ry(-0.7555593142697683) q[1];
ry(-2.882332768677802) q[2];
cx q[1],q[2];
ry(-2.132276762988758) q[1];
ry(-2.8052511894512397) q[3];
cx q[1],q[3];
ry(2.8199957251902252) q[1];
ry(-1.789534395330281) q[3];
cx q[1],q[3];
ry(-1.9356430700096223) q[1];
ry(2.3334366049468995) q[4];
cx q[1],q[4];
ry(-0.3763402506076056) q[1];
ry(1.4638474778721848) q[4];
cx q[1],q[4];
ry(-3.110891234077953) q[1];
ry(2.924117483629689) q[5];
cx q[1],q[5];
ry(-3.006779492087255) q[1];
ry(-0.5290875557559431) q[5];
cx q[1],q[5];
ry(-0.7646287397699051) q[1];
ry(0.9864084095336371) q[6];
cx q[1],q[6];
ry(1.6461404428964803) q[1];
ry(1.5466504729225707) q[6];
cx q[1],q[6];
ry(-2.6907109654167294) q[1];
ry(-2.2906318861874833) q[7];
cx q[1],q[7];
ry(0.8540360177779115) q[1];
ry(-0.9602704535626234) q[7];
cx q[1],q[7];
ry(-1.694924812337632) q[2];
ry(2.3513491048563604) q[3];
cx q[2],q[3];
ry(1.9832298757311777) q[2];
ry(1.5986153152813538) q[3];
cx q[2],q[3];
ry(-2.913131394993217) q[2];
ry(-1.67529550104941) q[4];
cx q[2],q[4];
ry(-2.9406671529893966) q[2];
ry(-1.4742527397433438) q[4];
cx q[2],q[4];
ry(-1.2721897750563569) q[2];
ry(-1.91201923858563) q[5];
cx q[2],q[5];
ry(0.3543913593686089) q[2];
ry(0.29682124526350046) q[5];
cx q[2],q[5];
ry(2.9676166976858225) q[2];
ry(-2.1836042106411364) q[6];
cx q[2],q[6];
ry(0.09822978064829363) q[2];
ry(-1.7037086795898944) q[6];
cx q[2],q[6];
ry(-0.6803665323206056) q[2];
ry(1.7408880232184405) q[7];
cx q[2],q[7];
ry(-1.7430777599167637) q[2];
ry(-2.2767655473443087) q[7];
cx q[2],q[7];
ry(-2.356904141292378) q[3];
ry(2.843684595879463) q[4];
cx q[3],q[4];
ry(0.24725761998149873) q[3];
ry(-0.5775456458745989) q[4];
cx q[3],q[4];
ry(2.498863439859469) q[3];
ry(-2.5576622603078207) q[5];
cx q[3],q[5];
ry(2.84771382347512) q[3];
ry(-0.28367647781502114) q[5];
cx q[3],q[5];
ry(-1.446785446790169) q[3];
ry(-2.036420971758935) q[6];
cx q[3],q[6];
ry(1.5173422058949537) q[3];
ry(-0.06536830595451182) q[6];
cx q[3],q[6];
ry(-1.1044073344055507) q[3];
ry(0.5017654517698364) q[7];
cx q[3],q[7];
ry(1.0699114985249096) q[3];
ry(-1.9624537336000705) q[7];
cx q[3],q[7];
ry(0.7005375532929001) q[4];
ry(-1.10316705723005) q[5];
cx q[4],q[5];
ry(0.08540857011773903) q[4];
ry(0.9442277102969227) q[5];
cx q[4],q[5];
ry(2.4410513958670377) q[4];
ry(0.43277562438913364) q[6];
cx q[4],q[6];
ry(-0.8811570852022781) q[4];
ry(-0.2562706864176132) q[6];
cx q[4],q[6];
ry(0.7467678768951496) q[4];
ry(2.718636480569427) q[7];
cx q[4],q[7];
ry(0.7962083995306958) q[4];
ry(1.4648838921799152) q[7];
cx q[4],q[7];
ry(0.4159941392532778) q[5];
ry(-0.24642232803455055) q[6];
cx q[5],q[6];
ry(-0.14486858383765888) q[5];
ry(-0.3063525705429546) q[6];
cx q[5],q[6];
ry(1.9950404574802567) q[5];
ry(2.733625189317224) q[7];
cx q[5],q[7];
ry(1.1637337614816596) q[5];
ry(-2.9403514434795177) q[7];
cx q[5],q[7];
ry(1.7571676429150358) q[6];
ry(1.712364968639456) q[7];
cx q[6],q[7];
ry(-2.0069078483933893) q[6];
ry(-2.722204911441343) q[7];
cx q[6],q[7];
ry(-1.1816060415230711) q[0];
ry(-1.0801175100988063) q[1];
cx q[0],q[1];
ry(-3.0735710162111998) q[0];
ry(-0.9933874939783625) q[1];
cx q[0],q[1];
ry(-3.1374605758157257) q[0];
ry(-2.442086863744417) q[2];
cx q[0],q[2];
ry(-2.623170045667909) q[0];
ry(-0.45709753309156687) q[2];
cx q[0],q[2];
ry(0.9952243752652706) q[0];
ry(-1.1084911845132632) q[3];
cx q[0],q[3];
ry(3.038072986783694) q[0];
ry(0.30928930625905693) q[3];
cx q[0],q[3];
ry(2.8913196471364104) q[0];
ry(-0.4801969590474865) q[4];
cx q[0],q[4];
ry(-0.38447907089018857) q[0];
ry(2.7707352755070946) q[4];
cx q[0],q[4];
ry(-1.9968259983830947) q[0];
ry(0.5851695875162095) q[5];
cx q[0],q[5];
ry(0.542362704339193) q[0];
ry(-2.8323597484866405) q[5];
cx q[0],q[5];
ry(3.0880230202106183) q[0];
ry(-2.4076327980100616) q[6];
cx q[0],q[6];
ry(-2.8237893252779553) q[0];
ry(1.1545008258643739) q[6];
cx q[0],q[6];
ry(1.0481551138190908) q[0];
ry(1.7148813670660896) q[7];
cx q[0],q[7];
ry(0.6018675268164078) q[0];
ry(-0.5525251838585351) q[7];
cx q[0],q[7];
ry(-0.6389858284924815) q[1];
ry(-2.6564468432568256) q[2];
cx q[1],q[2];
ry(-0.47623250162003816) q[1];
ry(-0.18536264806546235) q[2];
cx q[1],q[2];
ry(2.996176614287671) q[1];
ry(-1.6890437344054685) q[3];
cx q[1],q[3];
ry(-1.8894045630871033) q[1];
ry(-1.6886389383461904) q[3];
cx q[1],q[3];
ry(1.0102165866601036) q[1];
ry(-1.4479909005500917) q[4];
cx q[1],q[4];
ry(1.7469922857673037) q[1];
ry(-3.0328825623749216) q[4];
cx q[1],q[4];
ry(-1.5002000205067558) q[1];
ry(1.409641539666234) q[5];
cx q[1],q[5];
ry(-2.150545948370417) q[1];
ry(1.0844505616964615) q[5];
cx q[1],q[5];
ry(-2.172278074012722) q[1];
ry(2.6766668245579006) q[6];
cx q[1],q[6];
ry(1.3465571109851608) q[1];
ry(-0.31306781366959413) q[6];
cx q[1],q[6];
ry(1.8688061675064391) q[1];
ry(2.011751563772299) q[7];
cx q[1],q[7];
ry(2.2155605648853696) q[1];
ry(-0.8771519407291546) q[7];
cx q[1],q[7];
ry(-1.3363107154461904) q[2];
ry(-1.732345242923324) q[3];
cx q[2],q[3];
ry(2.521764151871233) q[2];
ry(2.883514061227184) q[3];
cx q[2],q[3];
ry(1.7333803287612757) q[2];
ry(-0.40662177338250327) q[4];
cx q[2],q[4];
ry(0.5079489368656764) q[2];
ry(1.2757994757061777) q[4];
cx q[2],q[4];
ry(-1.9578532501941668) q[2];
ry(2.8572274093738024) q[5];
cx q[2],q[5];
ry(-2.8169265470625326) q[2];
ry(-1.2160962779280498) q[5];
cx q[2],q[5];
ry(3.083996845878527) q[2];
ry(0.8639868873054155) q[6];
cx q[2],q[6];
ry(1.126262798004788) q[2];
ry(-2.9496054844312054) q[6];
cx q[2],q[6];
ry(0.0903510306872533) q[2];
ry(-2.0801028124447765) q[7];
cx q[2],q[7];
ry(2.0521619684155894) q[2];
ry(2.016363648116391) q[7];
cx q[2],q[7];
ry(-2.3218595597365677) q[3];
ry(2.60145606153462) q[4];
cx q[3],q[4];
ry(2.0187140430171686) q[3];
ry(-1.9208904057339642) q[4];
cx q[3],q[4];
ry(0.11537493943538024) q[3];
ry(-1.579750851478411) q[5];
cx q[3],q[5];
ry(2.863424750088648) q[3];
ry(1.3565595662406915) q[5];
cx q[3],q[5];
ry(0.2151620820106066) q[3];
ry(2.9459609034456142) q[6];
cx q[3],q[6];
ry(0.13269444729738072) q[3];
ry(2.612031121100519) q[6];
cx q[3],q[6];
ry(1.025137858621515) q[3];
ry(0.08071486212392821) q[7];
cx q[3],q[7];
ry(-1.0799242961638218) q[3];
ry(1.6155853355245746) q[7];
cx q[3],q[7];
ry(0.620537752724033) q[4];
ry(1.0109862277601405) q[5];
cx q[4],q[5];
ry(-1.1787330383369499) q[4];
ry(0.12627360041274294) q[5];
cx q[4],q[5];
ry(-0.6484568442573098) q[4];
ry(-1.411992576722783) q[6];
cx q[4],q[6];
ry(0.2771027861422066) q[4];
ry(-1.5657261956390385) q[6];
cx q[4],q[6];
ry(0.7377824534717926) q[4];
ry(-1.8232553214817937) q[7];
cx q[4],q[7];
ry(0.08318242569838308) q[4];
ry(1.6065741062227712) q[7];
cx q[4],q[7];
ry(-0.8644942900190555) q[5];
ry(-0.6404360425742099) q[6];
cx q[5],q[6];
ry(2.436441028660342) q[5];
ry(1.3535984586045062) q[6];
cx q[5],q[6];
ry(2.4615559191913166) q[5];
ry(2.7762997911412395) q[7];
cx q[5],q[7];
ry(2.7898335256740983) q[5];
ry(-1.0494605608268825) q[7];
cx q[5],q[7];
ry(-1.1469165073906007) q[6];
ry(1.717985930024159) q[7];
cx q[6],q[7];
ry(-2.929346804146291) q[6];
ry(2.142575695269638) q[7];
cx q[6],q[7];
ry(-1.9811139529717465) q[0];
ry(-0.9877738630385927) q[1];
cx q[0],q[1];
ry(1.3468606330159962) q[0];
ry(2.670357382989248) q[1];
cx q[0],q[1];
ry(-2.0576245333614374) q[0];
ry(-0.7656294646208863) q[2];
cx q[0],q[2];
ry(2.576483665243223) q[0];
ry(-0.5704333893712581) q[2];
cx q[0],q[2];
ry(2.2862645777615023) q[0];
ry(-2.880757176282235) q[3];
cx q[0],q[3];
ry(-2.6980709000815093) q[0];
ry(1.4567276422998132) q[3];
cx q[0],q[3];
ry(-0.8264760717049437) q[0];
ry(-2.6636360310028273) q[4];
cx q[0],q[4];
ry(-0.9902514372316955) q[0];
ry(-2.788565925512637) q[4];
cx q[0],q[4];
ry(-1.9351701550210332) q[0];
ry(-0.4003578892085171) q[5];
cx q[0],q[5];
ry(-0.8549361518038637) q[0];
ry(0.5660533343307828) q[5];
cx q[0],q[5];
ry(0.11716993140936025) q[0];
ry(1.5984179104691696) q[6];
cx q[0],q[6];
ry(-0.28727487307549904) q[0];
ry(-2.7851707420773826) q[6];
cx q[0],q[6];
ry(-0.27281331533370956) q[0];
ry(-0.6356515888784334) q[7];
cx q[0],q[7];
ry(0.5655414516510064) q[0];
ry(2.9227762300638305) q[7];
cx q[0],q[7];
ry(-2.7500193424651354) q[1];
ry(-1.9623111583175614) q[2];
cx q[1],q[2];
ry(-1.6187404440103856) q[1];
ry(-0.04526949614729623) q[2];
cx q[1],q[2];
ry(2.741654212139646) q[1];
ry(-1.7138679232472152) q[3];
cx q[1],q[3];
ry(-1.7613787295178431) q[1];
ry(2.756158139654249) q[3];
cx q[1],q[3];
ry(0.6092265525532783) q[1];
ry(1.903023593707446) q[4];
cx q[1],q[4];
ry(-1.0855304653350595) q[1];
ry(-1.718588614274232) q[4];
cx q[1],q[4];
ry(1.0941163894642076) q[1];
ry(2.105614023688585) q[5];
cx q[1],q[5];
ry(-0.3856643851056095) q[1];
ry(1.8852446123313529) q[5];
cx q[1],q[5];
ry(1.5873736093707795) q[1];
ry(0.5789828475352126) q[6];
cx q[1],q[6];
ry(2.497319097438288) q[1];
ry(1.273649855029757) q[6];
cx q[1],q[6];
ry(1.2418382187092352) q[1];
ry(2.839100845119185) q[7];
cx q[1],q[7];
ry(-2.138074334083789) q[1];
ry(-3.0476871050695045) q[7];
cx q[1],q[7];
ry(3.0075079362357275) q[2];
ry(2.1306348233890886) q[3];
cx q[2],q[3];
ry(-0.6549307162940923) q[2];
ry(-2.6017711345841463) q[3];
cx q[2],q[3];
ry(1.8427884925365015) q[2];
ry(-1.3236137338407827) q[4];
cx q[2],q[4];
ry(-1.3775440924088276) q[2];
ry(-2.8681571735918365) q[4];
cx q[2],q[4];
ry(-2.3746810581351334) q[2];
ry(-1.2808071050729533) q[5];
cx q[2],q[5];
ry(-1.3495927100733693) q[2];
ry(0.7981312646586589) q[5];
cx q[2],q[5];
ry(0.06823225732047404) q[2];
ry(-3.0765536561904594) q[6];
cx q[2],q[6];
ry(2.695195127333357) q[2];
ry(1.8460053001745689) q[6];
cx q[2],q[6];
ry(3.0779201128349283) q[2];
ry(-1.0356954996289154) q[7];
cx q[2],q[7];
ry(0.3065305816803221) q[2];
ry(-2.1371312265392506) q[7];
cx q[2],q[7];
ry(2.7350658668139016) q[3];
ry(0.6427724641898083) q[4];
cx q[3],q[4];
ry(0.44419638225042246) q[3];
ry(2.984941819400521) q[4];
cx q[3],q[4];
ry(-2.4668518395791255) q[3];
ry(2.522182513630993) q[5];
cx q[3],q[5];
ry(-0.5037715077638872) q[3];
ry(1.9400210231395423) q[5];
cx q[3],q[5];
ry(2.682260738490727) q[3];
ry(-0.5652425402577987) q[6];
cx q[3],q[6];
ry(2.1529125713608934) q[3];
ry(1.227608922675298) q[6];
cx q[3],q[6];
ry(2.613239389516347) q[3];
ry(1.6005551131937739) q[7];
cx q[3],q[7];
ry(-1.2994137446917864) q[3];
ry(-1.008049817481764) q[7];
cx q[3],q[7];
ry(0.9724140928640356) q[4];
ry(2.0817060411332093) q[5];
cx q[4],q[5];
ry(-0.563811422704567) q[4];
ry(1.3475910572960028) q[5];
cx q[4],q[5];
ry(1.1479817101451966) q[4];
ry(2.925146175901159) q[6];
cx q[4],q[6];
ry(-2.4207081704529245) q[4];
ry(0.8288259653308313) q[6];
cx q[4],q[6];
ry(-0.9718721384942227) q[4];
ry(3.1180697205675143) q[7];
cx q[4],q[7];
ry(-0.792251154464508) q[4];
ry(-0.017230504509424094) q[7];
cx q[4],q[7];
ry(2.264938650375291) q[5];
ry(1.9259366772254534) q[6];
cx q[5],q[6];
ry(-2.0159247680550534) q[5];
ry(-1.5402699304408936) q[6];
cx q[5],q[6];
ry(2.804202961642725) q[5];
ry(-1.4997283391589145) q[7];
cx q[5],q[7];
ry(1.8294125707334459) q[5];
ry(0.640460088356363) q[7];
cx q[5],q[7];
ry(-3.0457006860975855) q[6];
ry(0.9248968788811691) q[7];
cx q[6],q[7];
ry(-1.0129621972384002) q[6];
ry(-1.4758565470572238) q[7];
cx q[6],q[7];
ry(1.5099747678688509) q[0];
ry(-2.7836899754751485) q[1];
cx q[0],q[1];
ry(2.7926280649399953) q[0];
ry(2.999636044636122) q[1];
cx q[0],q[1];
ry(-2.212475232222308) q[0];
ry(-2.9091646765704717) q[2];
cx q[0],q[2];
ry(0.6987828533562421) q[0];
ry(0.001225396712807293) q[2];
cx q[0],q[2];
ry(1.1602535713051867) q[0];
ry(-0.704353726132291) q[3];
cx q[0],q[3];
ry(-1.8131984722948784) q[0];
ry(2.8417146964768456) q[3];
cx q[0],q[3];
ry(2.375265634229245) q[0];
ry(2.2210209579176405) q[4];
cx q[0],q[4];
ry(-1.6249049798528485) q[0];
ry(-0.5652328984650289) q[4];
cx q[0],q[4];
ry(2.7308611835830185) q[0];
ry(2.0074557583517034) q[5];
cx q[0],q[5];
ry(-1.090071349908893) q[0];
ry(0.8231782267331319) q[5];
cx q[0],q[5];
ry(-0.6480389732253297) q[0];
ry(-0.957873864330888) q[6];
cx q[0],q[6];
ry(2.8975087849208148) q[0];
ry(-1.377177069132763) q[6];
cx q[0],q[6];
ry(1.2326828785403299) q[0];
ry(-0.6536364172193407) q[7];
cx q[0],q[7];
ry(-1.8899138554314412) q[0];
ry(-2.6807866822045217) q[7];
cx q[0],q[7];
ry(-0.14525310631325453) q[1];
ry(0.7621442925695283) q[2];
cx q[1],q[2];
ry(1.5917684899042088) q[1];
ry(0.6409843787820098) q[2];
cx q[1],q[2];
ry(-2.697608067202498) q[1];
ry(2.028066269043113) q[3];
cx q[1],q[3];
ry(-1.4928271094660521) q[1];
ry(-0.5402479782604273) q[3];
cx q[1],q[3];
ry(-2.4271735945671393) q[1];
ry(-1.8939644309801151) q[4];
cx q[1],q[4];
ry(-3.0455841749913097) q[1];
ry(0.11804494581906155) q[4];
cx q[1],q[4];
ry(-2.3019479634926725) q[1];
ry(0.4981734813835974) q[5];
cx q[1],q[5];
ry(-1.6831171344373623) q[1];
ry(-1.7531750571902605) q[5];
cx q[1],q[5];
ry(1.9352357533651436) q[1];
ry(1.566764822585945) q[6];
cx q[1],q[6];
ry(0.8293708876795411) q[1];
ry(2.3194801469869213) q[6];
cx q[1],q[6];
ry(-2.892706157462418) q[1];
ry(0.5641986019546712) q[7];
cx q[1],q[7];
ry(-0.4487261602798034) q[1];
ry(-1.2080518422683264) q[7];
cx q[1],q[7];
ry(1.8814334596204212) q[2];
ry(1.7068798029055101) q[3];
cx q[2],q[3];
ry(0.47209864188428285) q[2];
ry(-1.0680176477486496) q[3];
cx q[2],q[3];
ry(-1.8307484255618265) q[2];
ry(2.5824355762773408) q[4];
cx q[2],q[4];
ry(2.340464615850953) q[2];
ry(1.7822955024518388) q[4];
cx q[2],q[4];
ry(-2.748844501753265) q[2];
ry(1.930008655887105) q[5];
cx q[2],q[5];
ry(-0.7043535275242873) q[2];
ry(-1.596616253103342) q[5];
cx q[2],q[5];
ry(-1.1588792139022823) q[2];
ry(2.986566124774243) q[6];
cx q[2],q[6];
ry(0.9615302370366703) q[2];
ry(1.7278131867292368) q[6];
cx q[2],q[6];
ry(-1.654117414561533) q[2];
ry(2.244716053644698) q[7];
cx q[2],q[7];
ry(-2.7243859335840743) q[2];
ry(2.7869066774592866) q[7];
cx q[2],q[7];
ry(0.31645730059928834) q[3];
ry(2.2411565220055794) q[4];
cx q[3],q[4];
ry(0.7513082978712115) q[3];
ry(-0.08944684642325562) q[4];
cx q[3],q[4];
ry(2.8177592141932766) q[3];
ry(1.5641320713608986) q[5];
cx q[3],q[5];
ry(0.736295574696129) q[3];
ry(-2.047273853537275) q[5];
cx q[3],q[5];
ry(0.24221587435835218) q[3];
ry(2.015048034200092) q[6];
cx q[3],q[6];
ry(-2.6268631782496814) q[3];
ry(-1.6265454981017138) q[6];
cx q[3],q[6];
ry(-1.6929314957575128) q[3];
ry(-0.3367898544097792) q[7];
cx q[3],q[7];
ry(-2.6014372856152184) q[3];
ry(1.9480325989224483) q[7];
cx q[3],q[7];
ry(-2.8450539283479928) q[4];
ry(-2.433345778144388) q[5];
cx q[4],q[5];
ry(-0.9260604473735627) q[4];
ry(2.8431884137183063) q[5];
cx q[4],q[5];
ry(-0.8306035807474469) q[4];
ry(-2.2993748772359632) q[6];
cx q[4],q[6];
ry(1.6523502409870618) q[4];
ry(1.061375444277979) q[6];
cx q[4],q[6];
ry(-2.7241754370269247) q[4];
ry(2.426788666311354) q[7];
cx q[4],q[7];
ry(2.739368471684784) q[4];
ry(-1.3877392158462243) q[7];
cx q[4],q[7];
ry(-2.10675060498554) q[5];
ry(0.30402656541660467) q[6];
cx q[5],q[6];
ry(0.4031744811095689) q[5];
ry(1.8676069191405935) q[6];
cx q[5],q[6];
ry(-1.9504158858231324) q[5];
ry(-3.0124226399952896) q[7];
cx q[5],q[7];
ry(2.0695944520031064) q[5];
ry(1.9933155754830896) q[7];
cx q[5],q[7];
ry(-1.5685435370029257) q[6];
ry(2.3711466461996933) q[7];
cx q[6],q[7];
ry(1.4473604820415806) q[6];
ry(1.7238338938116804) q[7];
cx q[6],q[7];
ry(0.5280436819279606) q[0];
ry(-2.0714631174348517) q[1];
cx q[0],q[1];
ry(-0.20803485112860187) q[0];
ry(1.8523565297956512) q[1];
cx q[0],q[1];
ry(2.6978336724813605) q[0];
ry(-0.9775855788430299) q[2];
cx q[0],q[2];
ry(-2.0240712676400534) q[0];
ry(1.16629104904815) q[2];
cx q[0],q[2];
ry(-1.7899313497772518) q[0];
ry(-2.6733830571509833) q[3];
cx q[0],q[3];
ry(-1.4615837956780258) q[0];
ry(0.7147517430835197) q[3];
cx q[0],q[3];
ry(-2.5446503642163565) q[0];
ry(-1.905779396031448) q[4];
cx q[0],q[4];
ry(-1.4984415706689713) q[0];
ry(-2.572014011380215) q[4];
cx q[0],q[4];
ry(-2.026811050036713) q[0];
ry(-2.3244528896880725) q[5];
cx q[0],q[5];
ry(-0.7415392131643701) q[0];
ry(-2.8356352698003287) q[5];
cx q[0],q[5];
ry(3.03165378499369) q[0];
ry(2.813909983260317) q[6];
cx q[0],q[6];
ry(-2.5159731789768545) q[0];
ry(-1.287300129137031) q[6];
cx q[0],q[6];
ry(-2.1844896213615375) q[0];
ry(1.2521699202658036) q[7];
cx q[0],q[7];
ry(1.7647883824868078) q[0];
ry(1.7567715818321705) q[7];
cx q[0],q[7];
ry(1.7155107203255868) q[1];
ry(2.1760835629077855) q[2];
cx q[1],q[2];
ry(-0.629250238617117) q[1];
ry(-0.08319741142204219) q[2];
cx q[1],q[2];
ry(1.5993810601707819) q[1];
ry(1.4105874631987938) q[3];
cx q[1],q[3];
ry(-0.49942384093183545) q[1];
ry(0.9912519229860396) q[3];
cx q[1],q[3];
ry(-2.7153432089512366) q[1];
ry(0.06421811414108802) q[4];
cx q[1],q[4];
ry(1.478717243082313) q[1];
ry(-3.025210240858426) q[4];
cx q[1],q[4];
ry(-2.5793252767704153) q[1];
ry(2.6301081735575518) q[5];
cx q[1],q[5];
ry(1.486819985188994) q[1];
ry(2.5168034228022407) q[5];
cx q[1],q[5];
ry(-0.06571852474337538) q[1];
ry(-2.667298768277708) q[6];
cx q[1],q[6];
ry(0.20344836091538365) q[1];
ry(-2.5641603715098196) q[6];
cx q[1],q[6];
ry(2.8043171573892005) q[1];
ry(0.3234566973950297) q[7];
cx q[1],q[7];
ry(-2.4201734257276613) q[1];
ry(-2.2621406351788353) q[7];
cx q[1],q[7];
ry(0.2746771394419644) q[2];
ry(1.0086368271767396) q[3];
cx q[2],q[3];
ry(-0.41926286019111725) q[2];
ry(-2.7862323366325428) q[3];
cx q[2],q[3];
ry(-1.6735413372133445) q[2];
ry(0.9529280027745771) q[4];
cx q[2],q[4];
ry(0.03371914661406059) q[2];
ry(0.10858422993715688) q[4];
cx q[2],q[4];
ry(-2.9512271131199634) q[2];
ry(2.2032204322305144) q[5];
cx q[2],q[5];
ry(0.051742850098060025) q[2];
ry(1.5119192267429398) q[5];
cx q[2],q[5];
ry(2.9734810130648768) q[2];
ry(3.0256141358198554) q[6];
cx q[2],q[6];
ry(-1.9810383809990322) q[2];
ry(0.913471352436816) q[6];
cx q[2],q[6];
ry(1.2925289732544623) q[2];
ry(2.048039432878043) q[7];
cx q[2],q[7];
ry(1.5502899828922425) q[2];
ry(2.9087233028280344) q[7];
cx q[2],q[7];
ry(-1.6390786752583255) q[3];
ry(0.28084412769465233) q[4];
cx q[3],q[4];
ry(2.164193839306262) q[3];
ry(-1.3264380088302667) q[4];
cx q[3],q[4];
ry(2.041554971734148) q[3];
ry(1.9197621896149781) q[5];
cx q[3],q[5];
ry(-0.16180497236630992) q[3];
ry(-0.14096705968706402) q[5];
cx q[3],q[5];
ry(-0.37987840545566554) q[3];
ry(2.264699739391227) q[6];
cx q[3],q[6];
ry(-2.0945589416765378) q[3];
ry(-0.09658505465199439) q[6];
cx q[3],q[6];
ry(3.063962747262445) q[3];
ry(0.8213319753147585) q[7];
cx q[3],q[7];
ry(1.6048017596877877) q[3];
ry(-2.7142495111570333) q[7];
cx q[3],q[7];
ry(-1.0229343207712158) q[4];
ry(-0.7478117561346408) q[5];
cx q[4],q[5];
ry(0.9348700298379748) q[4];
ry(2.903659193620339) q[5];
cx q[4],q[5];
ry(1.7364812929456148) q[4];
ry(1.8972630085998343) q[6];
cx q[4],q[6];
ry(1.3843416535315738) q[4];
ry(-1.8386991529733485) q[6];
cx q[4],q[6];
ry(1.0232544249379294) q[4];
ry(-2.9237321552647475) q[7];
cx q[4],q[7];
ry(0.6813792805274854) q[4];
ry(-2.875300770887977) q[7];
cx q[4],q[7];
ry(2.499313983287176) q[5];
ry(-0.5090446966513102) q[6];
cx q[5],q[6];
ry(0.7577411601301947) q[5];
ry(0.4805503407715319) q[6];
cx q[5],q[6];
ry(0.044323550648933875) q[5];
ry(-2.514336721620066) q[7];
cx q[5],q[7];
ry(2.4227392164927926) q[5];
ry(-0.11869529657840394) q[7];
cx q[5],q[7];
ry(1.7011106628730615) q[6];
ry(0.4615366128475777) q[7];
cx q[6],q[7];
ry(0.436274697970828) q[6];
ry(0.8742589138637931) q[7];
cx q[6],q[7];
ry(1.9114364756216418) q[0];
ry(0.19516284876524936) q[1];
cx q[0],q[1];
ry(-1.2856649491604941) q[0];
ry(-2.3525925757373183) q[1];
cx q[0],q[1];
ry(3.075623746078353) q[0];
ry(0.992945133125203) q[2];
cx q[0],q[2];
ry(2.0876876248398784) q[0];
ry(1.504340880554186) q[2];
cx q[0],q[2];
ry(0.2874460515431041) q[0];
ry(-1.3916322371960794) q[3];
cx q[0],q[3];
ry(-0.893969564709082) q[0];
ry(1.7755897550965702) q[3];
cx q[0],q[3];
ry(-1.3829803264341) q[0];
ry(-1.3121912857817524) q[4];
cx q[0],q[4];
ry(3.0597484945348876) q[0];
ry(0.21314407444590167) q[4];
cx q[0],q[4];
ry(-2.1752563528313376) q[0];
ry(-1.7785964740053997) q[5];
cx q[0],q[5];
ry(-0.18119658023152832) q[0];
ry(-1.6285441129811855) q[5];
cx q[0],q[5];
ry(-0.12687034538860686) q[0];
ry(0.11497012032970398) q[6];
cx q[0],q[6];
ry(1.9778981215973135) q[0];
ry(-2.302709578347311) q[6];
cx q[0],q[6];
ry(-0.7120660954204334) q[0];
ry(-0.8214022106195902) q[7];
cx q[0],q[7];
ry(-0.8629450871745356) q[0];
ry(0.5003001821155388) q[7];
cx q[0],q[7];
ry(-2.0450787337044796) q[1];
ry(0.2736744248936241) q[2];
cx q[1],q[2];
ry(-2.95172502087481) q[1];
ry(1.570281586948642) q[2];
cx q[1],q[2];
ry(-3.003857690653558) q[1];
ry(-1.1016570397182157) q[3];
cx q[1],q[3];
ry(2.873050820894222) q[1];
ry(2.920555443615547) q[3];
cx q[1],q[3];
ry(0.4425599844069161) q[1];
ry(-2.5055120570223455) q[4];
cx q[1],q[4];
ry(0.4485594182459616) q[1];
ry(2.4245095461837614) q[4];
cx q[1],q[4];
ry(-2.876938429190644) q[1];
ry(3.001379452486521) q[5];
cx q[1],q[5];
ry(-2.1332952535342136) q[1];
ry(-0.6429573081550768) q[5];
cx q[1],q[5];
ry(1.165074212762602) q[1];
ry(-1.3620909192507) q[6];
cx q[1],q[6];
ry(-0.6766816627740083) q[1];
ry(2.4436368963467996) q[6];
cx q[1],q[6];
ry(2.3928074953861067) q[1];
ry(1.078506995829655) q[7];
cx q[1],q[7];
ry(-1.9122652894745071) q[1];
ry(-1.6871395001732388) q[7];
cx q[1],q[7];
ry(-2.7887263414324965) q[2];
ry(-3.1268446036253437) q[3];
cx q[2],q[3];
ry(0.02188667561618426) q[2];
ry(2.695734888008661) q[3];
cx q[2],q[3];
ry(-2.446499163013181) q[2];
ry(-1.943146932212444) q[4];
cx q[2],q[4];
ry(0.8499228575959847) q[2];
ry(1.8156756746662441) q[4];
cx q[2],q[4];
ry(-2.4022838277357565) q[2];
ry(-2.357491669181358) q[5];
cx q[2],q[5];
ry(0.590353843241072) q[2];
ry(2.8885020776750885) q[5];
cx q[2],q[5];
ry(0.13873278730619276) q[2];
ry(1.656518333728835) q[6];
cx q[2],q[6];
ry(-2.483566798183223) q[2];
ry(-0.15056273407711598) q[6];
cx q[2],q[6];
ry(1.7752449108354642) q[2];
ry(-0.2409125747813437) q[7];
cx q[2],q[7];
ry(1.820924114236074) q[2];
ry(2.5660864632890963) q[7];
cx q[2],q[7];
ry(2.5326915594268264) q[3];
ry(-0.16408730284683962) q[4];
cx q[3],q[4];
ry(-1.3094126050304018) q[3];
ry(1.6042093093104255) q[4];
cx q[3],q[4];
ry(1.4109972185309987) q[3];
ry(0.7973157079349487) q[5];
cx q[3],q[5];
ry(2.88512998508223) q[3];
ry(-2.175366663118818) q[5];
cx q[3],q[5];
ry(-1.946873466472423) q[3];
ry(1.5668237932782214) q[6];
cx q[3],q[6];
ry(1.8926668909365345) q[3];
ry(-3.133157356905223) q[6];
cx q[3],q[6];
ry(-0.6207700240435083) q[3];
ry(-2.181299974597229) q[7];
cx q[3],q[7];
ry(-1.5862718635789566) q[3];
ry(0.09079612601640977) q[7];
cx q[3],q[7];
ry(-0.6289577254908775) q[4];
ry(-0.904120787229826) q[5];
cx q[4],q[5];
ry(-1.4278643974959915) q[4];
ry(-0.43721841760550806) q[5];
cx q[4],q[5];
ry(1.2326637816727262) q[4];
ry(-1.941033963090038) q[6];
cx q[4],q[6];
ry(-1.3861410583332878) q[4];
ry(-0.8946720121821174) q[6];
cx q[4],q[6];
ry(0.153112823065257) q[4];
ry(0.8633791998934371) q[7];
cx q[4],q[7];
ry(0.4019437296778552) q[4];
ry(0.9522465247936727) q[7];
cx q[4],q[7];
ry(-2.2595527696030278) q[5];
ry(2.1571042814795423) q[6];
cx q[5],q[6];
ry(2.918431442824428) q[5];
ry(2.4399065448902406) q[6];
cx q[5],q[6];
ry(-0.8295399074323667) q[5];
ry(3.087930448600435) q[7];
cx q[5],q[7];
ry(2.9817962548348715) q[5];
ry(-0.9679221499514504) q[7];
cx q[5],q[7];
ry(-2.5081019726476104) q[6];
ry(-1.953692641631607) q[7];
cx q[6],q[7];
ry(-1.8353635743167407) q[6];
ry(0.017892928755500165) q[7];
cx q[6],q[7];
ry(2.666395147983648) q[0];
ry(1.8637162058370622) q[1];
ry(-0.4763803672945304) q[2];
ry(2.1540717040091537) q[3];
ry(0.9754844896728851) q[4];
ry(0.6859030346123757) q[5];
ry(-2.644166548364847) q[6];
ry(-0.901672363421145) q[7];