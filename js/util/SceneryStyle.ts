// Copyright 2025, University of Colorado Boulder

/**
 * Creates and references a stylesheet that can be built up while Scenery is loading.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

// We may have to recreate the style element and stylesheet
let styleElement!: HTMLStyleElement;
let stylesheet!: CSSStyleSheet;

// We watch what rule strings are added, so we can reapply them if we need to recreate the stylesheet
const ruleStrings: string[] = [];

// Create a new element/stylesheet, and add it to the head
const attachStylesheet = (): void => {
  styleElement = document.createElement( 'style' );
  document.head.appendChild( styleElement );

  stylesheet = styleElement.sheet!;
  assert && assert( !stylesheet.disabled );
};
attachStylesheet();

const addRuleToStylesheet = ( ruleString: string ): void => {
  // using a this reference so it doesn't need to be a closure
  try {
    stylesheet.insertRule( ruleString, 0 );
  }
  catch( e ) {
    try {
      stylesheet.insertRule( ruleString, stylesheet.cssRules.length );
    }
    catch( e ) {
      console.log( 'Error adding CSS rule: ' + ruleString );
    }
  }
};

export const addCSSRule = ( ruleString: string ): void => {
  ruleStrings.push( ruleString );

  addRuleToStylesheet( ruleString );
};
scenery.register( 'addCSSRule', addCSSRule );

// Support for reapplying styles if they get wiped out (e.g. mkdocs material clearing our CSS).
// See https://github.com/scenerystack/community/issues/130
export const reapplyGlobalStyle = (): void => {
  if ( !document.head.contains( styleElement ) ) {
    attachStylesheet();

    ruleStrings.forEach( addRuleToStylesheet );
  }
};
scenery.register( 'reapplyGlobalStyle', reapplyGlobalStyle );