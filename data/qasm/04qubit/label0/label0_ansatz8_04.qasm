OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.2625429705091404) q[0];
ry(-1.650076144443534) q[1];
cx q[0],q[1];
ry(2.402647271936575) q[0];
ry(-2.5500017688023435) q[1];
cx q[0],q[1];
ry(-1.3654020610603248) q[2];
ry(1.8857466256745408) q[3];
cx q[2],q[3];
ry(0.4959073165801953) q[2];
ry(1.6850830844719429) q[3];
cx q[2],q[3];
ry(-1.7122459879227774) q[0];
ry(2.7381674637130784) q[2];
cx q[0],q[2];
ry(-2.415986340381493) q[0];
ry(2.3609212619728575) q[2];
cx q[0],q[2];
ry(-1.8267435007729025) q[1];
ry(1.193404829704995) q[3];
cx q[1],q[3];
ry(-0.07536325299542537) q[1];
ry(1.8113846191177023) q[3];
cx q[1],q[3];
ry(-2.281627113878457) q[0];
ry(2.462373105626585) q[1];
cx q[0],q[1];
ry(-0.5742537701273589) q[0];
ry(-1.844982508537869) q[1];
cx q[0],q[1];
ry(0.008833146570418124) q[2];
ry(2.9354250109375655) q[3];
cx q[2],q[3];
ry(-2.672970121866135) q[2];
ry(1.5675297029123267) q[3];
cx q[2],q[3];
ry(-0.3528959004467689) q[0];
ry(-1.5319986459646102) q[2];
cx q[0],q[2];
ry(0.7713925265390538) q[0];
ry(-0.9179308457386415) q[2];
cx q[0],q[2];
ry(2.976523953774129) q[1];
ry(1.0560789197036309) q[3];
cx q[1],q[3];
ry(-0.3202088273981536) q[1];
ry(-0.5106747552851935) q[3];
cx q[1],q[3];
ry(-2.428380400466916) q[0];
ry(0.04211567947306085) q[1];
cx q[0],q[1];
ry(0.5789736341251421) q[0];
ry(2.769028734631343) q[1];
cx q[0],q[1];
ry(-3.0368388766841377) q[2];
ry(-0.6262765429235818) q[3];
cx q[2],q[3];
ry(1.2913890146592064) q[2];
ry(1.7557000160450464) q[3];
cx q[2],q[3];
ry(1.3356887857532618) q[0];
ry(-1.0415128794811466) q[2];
cx q[0],q[2];
ry(-1.7566332809186445) q[0];
ry(-1.0509302453434621) q[2];
cx q[0],q[2];
ry(1.6974138747518062) q[1];
ry(-3.114217978030906) q[3];
cx q[1],q[3];
ry(-0.5281729207445797) q[1];
ry(3.0748181003795447) q[3];
cx q[1],q[3];
ry(2.899621900829994) q[0];
ry(-1.3504651566502515) q[1];
cx q[0],q[1];
ry(1.6046958808710285) q[0];
ry(-1.465279963578029) q[1];
cx q[0],q[1];
ry(1.0191672050356946) q[2];
ry(-1.7386343811336602) q[3];
cx q[2],q[3];
ry(-1.3628765493208197) q[2];
ry(-1.4634206615363423) q[3];
cx q[2],q[3];
ry(0.9561889903763845) q[0];
ry(0.46690810944379113) q[2];
cx q[0],q[2];
ry(1.6754279417792193) q[0];
ry(1.1435507083722236) q[2];
cx q[0],q[2];
ry(-1.4611529412636628) q[1];
ry(0.2275892868264678) q[3];
cx q[1],q[3];
ry(0.06295191672604172) q[1];
ry(-1.6746050904779208) q[3];
cx q[1],q[3];
ry(-1.035778876234363) q[0];
ry(0.14691438141907284) q[1];
cx q[0],q[1];
ry(2.62482355043839) q[0];
ry(-1.72747689477473) q[1];
cx q[0],q[1];
ry(3.001462147827446) q[2];
ry(0.053042625516472874) q[3];
cx q[2],q[3];
ry(1.839001151452635) q[2];
ry(-2.2084714893180557) q[3];
cx q[2],q[3];
ry(-2.2263532892060933) q[0];
ry(2.538619966283693) q[2];
cx q[0],q[2];
ry(0.28968682478928803) q[0];
ry(0.9838845678396559) q[2];
cx q[0],q[2];
ry(-0.9817072856778237) q[1];
ry(-1.752904955055653) q[3];
cx q[1],q[3];
ry(-0.10274767966182807) q[1];
ry(2.7274906240735994) q[3];
cx q[1],q[3];
ry(2.065784091797255) q[0];
ry(-0.514210790276846) q[1];
cx q[0],q[1];
ry(-1.480151958020806) q[0];
ry(1.8985403477792566) q[1];
cx q[0],q[1];
ry(-0.3581308299940575) q[2];
ry(-2.167212693437814) q[3];
cx q[2],q[3];
ry(-2.1443657422578326) q[2];
ry(0.5634592871325941) q[3];
cx q[2],q[3];
ry(2.540698894915363) q[0];
ry(-2.1487090600725143) q[2];
cx q[0],q[2];
ry(1.3607949574121017) q[0];
ry(-2.9900620373874753) q[2];
cx q[0],q[2];
ry(0.6390180842653379) q[1];
ry(-0.123915876117624) q[3];
cx q[1],q[3];
ry(0.35172987508426967) q[1];
ry(-0.07935479043289319) q[3];
cx q[1],q[3];
ry(1.3428281016561083) q[0];
ry(1.7753968114397436) q[1];
cx q[0],q[1];
ry(2.905653063483179) q[0];
ry(-1.2574394884852687) q[1];
cx q[0],q[1];
ry(-1.8505180171236835) q[2];
ry(1.9378446465266173) q[3];
cx q[2],q[3];
ry(2.8030274728916904) q[2];
ry(-1.5096307114510836) q[3];
cx q[2],q[3];
ry(0.9221165037810408) q[0];
ry(1.5973078257355444) q[2];
cx q[0],q[2];
ry(-1.835903998936801) q[0];
ry(-0.3967452765521669) q[2];
cx q[0],q[2];
ry(2.829517253725707) q[1];
ry(1.3064927854931234) q[3];
cx q[1],q[3];
ry(2.328352411579877) q[1];
ry(-3.0610246279009585) q[3];
cx q[1],q[3];
ry(0.2934065565376295) q[0];
ry(-1.991619120120657) q[1];
ry(-0.3871631978299206) q[2];
ry(2.0410971999459866) q[3];