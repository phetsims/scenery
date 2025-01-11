// Copyright 2013-2025, University of Colorado Boulder

/**
 * Creates and references a stylesheet that can be built up while Scenery is loading.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../imports.js';

const styleElement = document.createElement( 'style' );
styleElement.type = 'text/css';
document.head.appendChild( styleElement );

const stylesheet = document.styleSheets[ document.styleSheets.length - 1 ];
assert && assert( stylesheet.disabled === false );

const SceneryStyle = {
  stylesheet: stylesheet,
  styleElement: styleElement,

  addRule( ruleString ) {
    // using a this reference so it doesn't need to be a closure
    try {
      this.stylesheet.insertRule( ruleString, 0 );
    }
    catch( e ) {
      try {
        this.stylesheet.insertRule( ruleString, stylesheet.cssRules.length );
      }
      catch( e ) {
        console.log( 'Error adding CSS rule: ' + ruleString );
      }
    }
  }
};
scenery.register( 'SceneryStyle', SceneryStyle );
export default SceneryStyle;