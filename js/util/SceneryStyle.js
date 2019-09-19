// Copyright 2013-2015, University of Colorado Boulder


/**
 * Creates and references a stylesheet that can be built up while Scenery is loading.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const scenery = require( 'SCENERY/scenery' );

  const styleElement = document.createElement( 'style' );
  styleElement.type = 'text/css';
  document.head.appendChild( styleElement );

  const stylesheet = document.styleSheets[ document.styleSheets.length - 1 ];
  assert && assert( stylesheet.disabled === false );

  return scenery.register( 'SceneryStyle', {
    stylesheet: stylesheet,
    styleElement: styleElement,

    addRule: function( ruleString ) {
      // using a this reference so it doesn't need to be a closure
      this.stylesheet.insertRule( ruleString, 0 );
    }
  } );
} );
