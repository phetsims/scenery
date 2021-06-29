// Copyright 2016, University of Colorado Boulder

// WARNING: running this on a server is a MAJOR security risk, since it could allow arbitrary file reads/writes to a remote attacker.
// Only run this sporadically, and when behind a NAT router or firewall where it will never be touched.

// This is intended to allow reads/writes to the recorded sim data area (../data), depending on whether it is a GET or POST request

/* eslint-env node */

const http = require( 'http' );
const fs = require( 'fs' );

const ip = 'localhost'; // hardcoded for now, maybe 'localhost' will work?
const port = 8083;

http.createServer( ( req, res ) => {
  

  // see http://nodejs.org/api/http.html#http_request_method for docs

  const headers = {
    'Content-Type': 'text/plain',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'POST,OPTIONS'
  };

  // bail out quickly if it is just an OPTIONS request from CORS (we are on a different port, so allow cross-origin requests)
  if ( req.method === 'OPTIONS' ) {
    console.log( 'OPTIONS request' );
    res.writeHead( 200, headers );
    res.end( 'Success' );
    return;
  }

  const logname = decodeURIComponent( req.url.slice( 1 ) );
  if ( logname.indexOf( '..' ) !== -1 ) { throw new Error( `bad logname: ${logname}` ); } // brief check, there are probably other bad ways
  const logfile = `../data/${logname}.js`;

  // interpret a POST as a file write
  if ( req.method === 'POST' ) {
    let postdata = '';
    req.on( 'data', chunk => {
      postdata += chunk;
    } );
    req.on( 'end', () => {
      // logname = decodeURIComponent( req.url.slice( 1 ) );
      console.log( `write to logfile: ${logfile}` );

      fs.writeFile( logfile, postdata, err => {
        if ( err ) {
          console.log( err );

          res.writeHead( 500, headers );
          res.end( 'Failure' );
        }
        else {
          // console.log( "Saved:\n" + postdata );

          res.writeHead( 200, headers );
          res.end( 'Success' );
        }
      } );
    } );
    return;
  }

  // interpret a GET as a file read
  if ( req.method === 'GET' ) {
    fs.readFile( logfile, ( err, data ) => {
      if ( err ) {
        res.writeHead( 500, headers );
        res.end( 'Failure' );
      }
      else {
        res.writeHead( 200, headers );
        res.end( data );
      }
    } );
  }
} ).listen( port );
console.log( `ip: ${ip}` );
console.log( `port: ${port}` );
