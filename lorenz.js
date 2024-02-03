
var canvas = document.getElementById("myCanvas");
var ctx = canvas.getContext("2d");
var style = "#000000AA";

var treset = 1500;
var zscale = 4;
var xscale = 11;
var dt = 0.01;


// Set canvas size to window size
canvas.width = window.innerWidth;
canvas.height = 200;

if (canvas.width < 500) {
    var divscale = 0.5;
} else {
    var divscale = 0.8;
}

// Set line width
ctx.lineWidth = 1.0;
ctx.strokeStyle = style;

var s = 10;
var b = 8 / 3;
var r = 28;
function lorenz(x, y, z) {
    // https://en.wikipedia.org/wiki/Lorenz_system
    let dx = s * (y - x);
    let dy = x * (r - z) - y;
    let dz = x * y - b * z;
    return [dx, dy, dz];
}

var vmin = -5;
var vmax = 5;
function random_start() {
    return vmin + (vmax - vmin) * Math.random();
}

function repos_x(x) {
    return x * xscale + canvas.width * divscale;
}


function repos_z(z) {
    return z * zscale;
}


// Starting position
var x = random_start();
var y = random_start();
var z = random_start();

var xpaint = repos_x(x);
var zpaint = repos_z(z);


var i = 0;
setInterval(function () {
    ctx.beginPath();
    ctx.moveTo(xpaint, zpaint);
    let [dx, dy, dz] = lorenz(x, y, z);
    x += dx * dt;
    y += dy * dt;
    z += dz * dt;

    xpaint = repos_x(x);
    zpaint =  repos_z(z);
    ctx.lineTo(xpaint, zpaint);
    ctx.strokeStyle = style;
    ctx.stroke();

    if (i % treset == 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    i++;
}, 1);