OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-3.0559146166385758) q[0];
rz(2.342425094730956) q[0];
ry(2.6554811151915) q[1];
rz(-1.5637466069240835) q[1];
ry(-2.7884898308213613) q[2];
rz(-0.35780710544784133) q[2];
ry(-0.2176659753113359) q[3];
rz(-2.516901980377786) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6458079563345063) q[0];
rz(1.7038011796104233) q[0];
ry(0.8806457825955789) q[1];
rz(2.0957572557237247) q[1];
ry(-0.872376495726) q[2];
rz(0.5363604696767384) q[2];
ry(0.3202393317198112) q[3];
rz(-0.925589192751704) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.284965626770101) q[0];
rz(1.3853302098430316) q[0];
ry(-1.7944669235194792) q[1];
rz(-2.0941178566387015) q[1];
ry(2.2097085587010152) q[2];
rz(2.104433792924387) q[2];
ry(0.4360683222492012) q[3];
rz(0.6927494357069879) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.901270538574781) q[0];
rz(-1.9936053557328242) q[0];
ry(-2.183767540721301) q[1];
rz(-3.0937467973525026) q[1];
ry(-0.20394758180694783) q[2];
rz(-2.5988743096869467) q[2];
ry(2.142352895495422) q[3];
rz(2.045467679949186) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.9856566643685598) q[0];
rz(2.8460073579007155) q[0];
ry(-2.3483775495984833) q[1];
rz(-0.9679288680388749) q[1];
ry(-0.3693584511787364) q[2];
rz(-2.424667483363933) q[2];
ry(-1.1448577668827804) q[3];
rz(-2.2501604316952517) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.3502492347165944) q[0];
rz(1.6342827944634062) q[0];
ry(-0.9159242353856215) q[1];
rz(-2.6362289039445774) q[1];
ry(-0.48864216710389374) q[2];
rz(1.1993116657059941) q[2];
ry(-3.018810693979782) q[3];
rz(1.8294667039174541) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.043933201389022) q[0];
rz(1.3530337848014642) q[0];
ry(-0.45489831276309456) q[1];
rz(0.20894120015352605) q[1];
ry(-2.2767984441915763) q[2];
rz(-1.9099138614279694) q[2];
ry(-0.050065614639686826) q[3];
rz(3.056322044820044) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.7680754295907577) q[0];
rz(-1.2638313757132762) q[0];
ry(-2.690607847516853) q[1];
rz(-1.3400467250842514) q[1];
ry(-1.6055638240947887) q[2];
rz(-0.5064714578051425) q[2];
ry(0.8822474478554289) q[3];
rz(1.901969213186616) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.3509803392097579) q[0];
rz(-0.9421836390767683) q[0];
ry(1.470848389881179) q[1];
rz(-2.935413158423333) q[1];
ry(-2.578651203776642) q[2];
rz(-1.890601302403514) q[2];
ry(-2.3850530075090384) q[3];
rz(-2.083049457219926) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6470796967767727) q[0];
rz(1.6727651390134497) q[0];
ry(-1.0236987644255242) q[1];
rz(1.1927547392579727) q[1];
ry(0.978998531688204) q[2];
rz(0.664780911011396) q[2];
ry(0.17086606849448052) q[3];
rz(1.417853688506895) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.1225752624072864) q[0];
rz(0.24691613504088597) q[0];
ry(1.6969154757764453) q[1];
rz(2.182495851343396) q[1];
ry(-2.834084269284055) q[2];
rz(-0.2097538776215262) q[2];
ry(2.9291453396848546) q[3];
rz(0.7143661674720319) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9179188304731518) q[0];
rz(2.046711467230194) q[0];
ry(2.9783727558962827) q[1];
rz(3.132549500594621) q[1];
ry(-2.701214883190633) q[2];
rz(-3.0108987286682103) q[2];
ry(1.476248307388154) q[3];
rz(2.9897749322593663) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.8830153100932144) q[0];
rz(-2.4346751664133253) q[0];
ry(-2.8514287930167073) q[1];
rz(1.9391183426833543) q[1];
ry(2.8902664477035023) q[2];
rz(1.3084093522328568) q[2];
ry(0.3204945689986906) q[3];
rz(-0.5326105557308906) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.9489771267313385) q[0];
rz(2.1040765554128837) q[0];
ry(1.832000057216028) q[1];
rz(2.4615894508252048) q[1];
ry(0.8990583286701163) q[2];
rz(2.737491302929282) q[2];
ry(-2.91953992101847) q[3];
rz(0.15501605584958522) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.1242768547601214) q[0];
rz(-2.197457882894594) q[0];
ry(-0.7200871355873311) q[1];
rz(-1.8810950021804915) q[1];
ry(2.7209292514388728) q[2];
rz(1.4980338043855932) q[2];
ry(-1.5332478868601063) q[3];
rz(-2.5407574836800286) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.002884115376468) q[0];
rz(2.2954797528912283) q[0];
ry(2.7033085641484313) q[1];
rz(0.7881768638622094) q[1];
ry(0.08684757280965894) q[2];
rz(0.7583457032352202) q[2];
ry(-0.3180144949403453) q[3];
rz(-1.8601134945524578) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.0283484757935426) q[0];
rz(0.680037583746615) q[0];
ry(1.9330705778466744) q[1];
rz(2.9075389500970665) q[1];
ry(0.3217000134806242) q[2];
rz(2.347678015889716) q[2];
ry(2.7365522912776488) q[3];
rz(-2.9171941315908207) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.21260336830923) q[0];
rz(-1.2208488884366309) q[0];
ry(2.8519660679115573) q[1];
rz(-0.08330612799509647) q[1];
ry(-0.6038361313981306) q[2];
rz(1.9926089844284938) q[2];
ry(3.138101289482872) q[3];
rz(0.8129498349719936) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6592379661204779) q[0];
rz(0.1295498014027503) q[0];
ry(0.9081439803251694) q[1];
rz(-2.239499674977198) q[1];
ry(0.7173108944439219) q[2];
rz(-2.217448319060165) q[2];
ry(0.5062997406524838) q[3];
rz(-1.8487726964119198) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.556776520012856) q[0];
rz(-2.6661629695069355) q[0];
ry(-2.5655230981153347) q[1];
rz(-0.06918693335157398) q[1];
ry(3.0449529947967204) q[2];
rz(1.14348344225166) q[2];
ry(0.7501678548656496) q[3];
rz(0.4592433079648428) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.4631146232708026) q[0];
rz(-1.3785149697670045) q[0];
ry(-2.9725282932889816) q[1];
rz(1.2078933976901285) q[1];
ry(1.1336137496692835) q[2];
rz(-1.684051118760758) q[2];
ry(-1.7688109239083143) q[3];
rz(2.763638625635121) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.2581258509073678) q[0];
rz(-0.5954479623859897) q[0];
ry(1.3464355191564596) q[1];
rz(0.8600033335942728) q[1];
ry(1.1154165277426975) q[2];
rz(-0.7833786545849767) q[2];
ry(-1.6523103017534913) q[3];
rz(-1.4249591483113084) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.210519687230765) q[0];
rz(-1.7138462536971373) q[0];
ry(0.1371595726173266) q[1];
rz(2.3768830096297036) q[1];
ry(-2.204611970865341) q[2];
rz(3.0999190789354643) q[2];
ry(-2.8272207133493024) q[3];
rz(-3.10294407866753) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.1622181982452071) q[0];
rz(0.5325978271845901) q[0];
ry(0.07721177241956402) q[1];
rz(-1.0525283055869057) q[1];
ry(1.9890983536783873) q[2];
rz(-1.0084140646113526) q[2];
ry(-0.45667787546018973) q[3];
rz(0.8602759832052698) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.2556242949759209) q[0];
rz(0.5303245743902021) q[0];
ry(2.9361799211746193) q[1];
rz(1.998052588925355) q[1];
ry(-0.11343621297636891) q[2];
rz(2.0243723653637122) q[2];
ry(0.44484720639647984) q[3];
rz(-2.0330905718408236) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.0238933149504033) q[0];
rz(1.1275395364800191) q[0];
ry(-2.7298758173063953) q[1];
rz(2.944174876602767) q[1];
ry(2.701625464207754) q[2];
rz(0.4261047603496433) q[2];
ry(1.0958805913019143) q[3];
rz(-1.6871656837362075) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.6653826819705353) q[0];
rz(0.39471490963996725) q[0];
ry(2.263119800624933) q[1];
rz(2.8891030494712955) q[1];
ry(-1.288903174819393) q[2];
rz(-1.5327820175171578) q[2];
ry(-0.24586196105984562) q[3];
rz(2.1367918145227573) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.351419278097788) q[0];
rz(-2.333165393635121) q[0];
ry(-1.9437404211643878) q[1];
rz(-0.2865815327447184) q[1];
ry(-0.984428513050764) q[2];
rz(2.1317734764878398) q[2];
ry(-0.11810626525450349) q[3];
rz(1.9456493441741631) q[3];