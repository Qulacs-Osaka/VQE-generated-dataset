OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.1272161975337891) q[0];
rz(-2.875008903884337) q[0];
ry(-0.12894825379369168) q[1];
rz(1.4654782658168504) q[1];
ry(-2.3680683689176463) q[2];
rz(-2.1434604707330385) q[2];
ry(1.777084088643163) q[3];
rz(-0.9251735882226668) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.799684498768658) q[0];
rz(-2.802918777538137) q[0];
ry(-0.14187642427102087) q[1];
rz(-2.7003404923213474) q[1];
ry(0.8526371493744801) q[2];
rz(0.5799532197588071) q[2];
ry(-2.6065753056405936) q[3];
rz(2.0878192933089164) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.8198391433682044) q[0];
rz(-0.8479084593165531) q[0];
ry(-0.36994656444369384) q[1];
rz(-1.7102107732266187) q[1];
ry(-3.076533953118534) q[2];
rz(2.0034186441382245) q[2];
ry(-0.5460502614989649) q[3];
rz(1.5663302273667243) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4802984034910942) q[0];
rz(-2.155777128483674) q[0];
ry(2.6240229737472474) q[1];
rz(-1.930588212705202) q[1];
ry(-2.626206895432249) q[2];
rz(-1.5241137131162779) q[2];
ry(-1.6582102258237148) q[3];
rz(0.12925195307894913) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7985628021138842) q[0];
rz(0.5346277170815963) q[0];
ry(-1.0855305001640045) q[1];
rz(2.335863501684112) q[1];
ry(-1.7721103501688704) q[2];
rz(-2.358531246855095) q[2];
ry(1.9278648204469846) q[3];
rz(1.854369595023344) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.053629517106146) q[0];
rz(2.0684341623070974) q[0];
ry(0.9859969753688801) q[1];
rz(2.924280335596447) q[1];
ry(1.7942840514158238) q[2];
rz(-1.1842882018932972) q[2];
ry(2.090103734144635) q[3];
rz(-1.7679388844768207) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.0022059926802091567) q[0];
rz(1.7215759780645927) q[0];
ry(2.2506555843205636) q[1];
rz(-0.02825480880702768) q[1];
ry(-1.9811009055692912) q[2];
rz(2.449346279285768) q[2];
ry(-1.510176753982661) q[3];
rz(2.328649114537868) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.1222296619808954) q[0];
rz(1.900984539131663) q[0];
ry(-2.957054164448725) q[1];
rz(3.097721541408538) q[1];
ry(2.8920189668723255) q[2];
rz(-0.8731613123989348) q[2];
ry(-2.259337373749278) q[3];
rz(-3.0876753594327515) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.317504461545644) q[0];
rz(0.8743675282843187) q[0];
ry(-1.1944179614124786) q[1];
rz(-2.62394108631904) q[1];
ry(-2.2915337910836917) q[2];
rz(0.6763706914081704) q[2];
ry(-2.098810777315925) q[3];
rz(1.079944089736287) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3641843080664158) q[0];
rz(2.3398441530179905) q[0];
ry(-1.6858455533187477) q[1];
rz(3.049738440068846) q[1];
ry(1.9304794150314633) q[2];
rz(3.086662096400655) q[2];
ry(0.055574322139169574) q[3];
rz(0.5649635540988771) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7347223274878805) q[0];
rz(2.8997855550024356) q[0];
ry(-2.7951059805178207) q[1];
rz(0.6786684319769075) q[1];
ry(-0.2711092573608584) q[2];
rz(-2.604078856810552) q[2];
ry(-0.040260866464980616) q[3];
rz(1.3801319854574183) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.058625607168793) q[0];
rz(-0.45597162726732154) q[0];
ry(-2.0013035464725704) q[1];
rz(-0.5610584372425714) q[1];
ry(0.9189327237783889) q[2];
rz(-2.903113623450427) q[2];
ry(-2.356591365598904) q[3];
rz(0.030058758229595004) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.419907709171726) q[0];
rz(-2.660277778970399) q[0];
ry(-0.3413420859255409) q[1];
rz(-0.1651084188014451) q[1];
ry(-0.9047780692495984) q[2];
rz(-2.029751185313253) q[2];
ry(-2.647831849803219) q[3];
rz(-2.2686016032614713) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7996219002799281) q[0];
rz(0.9840159623691342) q[0];
ry(-2.2041985746892507) q[1];
rz(1.4457802375491902) q[1];
ry(0.22619133087904983) q[2];
rz(-1.6295875190593658) q[2];
ry(0.4235320088095298) q[3];
rz(0.28508420912262405) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8631251917052079) q[0];
rz(-2.798076912051927) q[0];
ry(-1.7323409640542602) q[1];
rz(-1.2816413290614737) q[1];
ry(0.6179631450259978) q[2];
rz(2.3554941721948732) q[2];
ry(-1.4621881837180786) q[3];
rz(2.1975537565763785) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5655656973651313) q[0];
rz(-0.05989659120931535) q[0];
ry(0.4134681270172811) q[1];
rz(-0.8096207965904504) q[1];
ry(1.4349077090445626) q[2];
rz(-1.1468825413060306) q[2];
ry(-1.4627855375816399) q[3];
rz(0.451883978112669) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.736076079608544) q[0];
rz(-3.0714987361329316) q[0];
ry(3.084608991507689) q[1];
rz(-1.8252962546474547) q[1];
ry(-1.6416237978838) q[2];
rz(-2.5467951100594735) q[2];
ry(1.0739895576912923) q[3];
rz(-0.2580695433110076) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.2846872774871336) q[0];
rz(-1.3959420179693174) q[0];
ry(0.736838522307854) q[1];
rz(-0.008334701829319828) q[1];
ry(-2.229525611328075) q[2];
rz(-2.490833257040709) q[2];
ry(3.1305284106517735) q[3];
rz(0.24822572892854156) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.553645259612481) q[0];
rz(0.825357982208584) q[0];
ry(1.434883618988765) q[1];
rz(-0.41374476352701406) q[1];
ry(0.39009782712361396) q[2];
rz(0.5495153493655881) q[2];
ry(-2.3932509079793625) q[3];
rz(-2.835166686162529) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.934598409197673) q[0];
rz(1.6533019536866878) q[0];
ry(-1.0343505859830289) q[1];
rz(3.0285119086292744) q[1];
ry(-2.255700353371785) q[2];
rz(3.1282235584734246) q[2];
ry(0.709551470563162) q[3];
rz(-2.8723382973666283) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1025227666951047) q[0];
rz(-0.09651554960929158) q[0];
ry(0.009266857297715703) q[1];
rz(2.4170268147433456) q[1];
ry(1.8294012050113633) q[2];
rz(0.897712014958412) q[2];
ry(0.2990002408755866) q[3];
rz(2.473218705799561) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.5850386249590178) q[0];
rz(-2.849280933985126) q[0];
ry(-2.6759261864226143) q[1];
rz(-0.07583611081146167) q[1];
ry(-1.1593535200679623) q[2];
rz(2.6978122970682628) q[2];
ry(-1.3563748552528079) q[3];
rz(-1.409300214672018) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.390631084165367) q[0];
rz(0.6137618618065617) q[0];
ry(1.3715875425447301) q[1];
rz(-2.5890624332406573) q[1];
ry(-0.5083052582539507) q[2];
rz(1.9384838862000286) q[2];
ry(1.5454026718080272) q[3];
rz(0.04578997146456665) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3515592805435874) q[0];
rz(1.9093074702650759) q[0];
ry(-2.1824361221831614) q[1];
rz(-1.0520224444776245) q[1];
ry(0.6211707590577266) q[2];
rz(-0.11622945200413515) q[2];
ry(-0.00021949919356067937) q[3];
rz(2.7857190770229505) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2991838345768612) q[0];
rz(-1.1305966189452747) q[0];
ry(-2.651205281742907) q[1];
rz(3.1073232191095235) q[1];
ry(-3.0984511369579764) q[2];
rz(-0.8152488837888562) q[2];
ry(0.646614862802843) q[3];
rz(2.214404486431449) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.4441540081818447) q[0];
rz(-2.200989235547544) q[0];
ry(-1.5695462735145145) q[1];
rz(0.33453625426237166) q[1];
ry(-0.2298397758110431) q[2];
rz(-3.021654825441485) q[2];
ry(-2.150436484147449) q[3];
rz(-1.8530921770315851) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7463528548513263) q[0];
rz(1.0324213407561604) q[0];
ry(1.4255274296932334) q[1];
rz(1.8729222827902794) q[1];
ry(-2.935293902242539) q[2];
rz(2.809324608588165) q[2];
ry(2.792474544299944) q[3];
rz(-0.17782131777677676) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7334163223357981) q[0];
rz(-2.40046839772937) q[0];
ry(-1.132379792519762) q[1];
rz(-1.9871540049631404) q[1];
ry(0.4553909857460094) q[2];
rz(-0.802740488942347) q[2];
ry(-1.323626605049614) q[3];
rz(-1.4437928524191896) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.5196929332097576) q[0];
rz(0.12236124152323401) q[0];
ry(-2.757098618664065) q[1];
rz(1.1586444791065205) q[1];
ry(-1.977672659548277) q[2];
rz(2.391702081566937) q[2];
ry(-1.2584664169572006) q[3];
rz(1.9114538483240866) q[3];