// Copyright 2019-2021, University of Colorado Boulder

/**
 * Static CSS style for elements of the PDOM (siblings of PDOMPeer). Adds the styling directly to SceneryStyle,
 * but also exports the class names for root and siblings for where elements are created or retrieved
 * by document.getElementsByClassName().
 *
 * REVIEW: Rename this to AccessiblePDOMStyle (or something more universal)
 *   - For all usages of "sibling" in this file.
 *
 * @author Jesse Greenberg
 */

import { scenery, SceneryStyle } from '../../imports.js';

// constants
const SIBLING_CLASS_NAME = 'a11y-pdom-element';
const ROOT_CLASS_NAME = 'a11y-pdom-root';
const LIST_ITEM_CLASS_NAME = 'a11y-pdom-list-item';

// All elements that use PDOMUtils.createElement should have this style. The only exception is the root of
// the PDOM, which should use root class attributes instead.
SceneryStyle.addRule( `.${SIBLING_CLASS_NAME
                      }{` +

                      // 'fixed' positions elements relative to the ViewPort (global coordinate frame), a requirement for the approach
                      // in PDOMPeer.positionElements
                      'position: fixed;' +

                      // ABSOLUTELY CRITICAL - so PDOM elements do not interfere with rest of scenery input
                      'pointer-events: none;' +

                      // default, to the 'relative' root PDOM element - will change with node transform if focusable
                      'top: 0px;' +
                      'left: 0px;' +

                      // for CSS transformations of focusable elements, origin at left top
                      'transform-origin: left top 0px;' +

                      // helps get accurate bounds with getBoundingClientRect() for transformations
                      'border-width: 0px;' +
                      'border: 0px;' +
                      'padding: 1px 1px;' + // cannot be zero, otherwise certain elements will have undefined width and height
                      'margin: 0px;' +
                      'white-space: nowrap;' +

                      // to remove the default focus highlight around HTML elements
                      'outline: none;' +
                      'box-shadow:none;' +
                      'border-color:transparent;' +

                      // So that elements can never be seen visually, can comment this out to "see" transformed elements in the
                      // PDOM. Text is made very small so that it doesn't extend into the display. Very low opacity on the root takes care of the rest.
                      'font-size: 1px;' + // must be at least 1px to be readable with AT

                      // adding this clip area seems to prevent Safari from doing expensive DOM layout calculations every change.
                      // Surprisingly, there is no performance benefit if this is put on the root element.
                      // see https://github.com/phetsims/scenery/issues/663
                      'clip: rect(1px, 1px, 1px, 1px);' +
                      '}'
);

SceneryStyle.addRule( `.${ROOT_CLASS_NAME
                      }{` +
                      // so that this root can also be positioned
                      'position: absolute;' +

                      // 'fixed' positioned elements interfere with workarounds that are meant to prevent Safari from going to sleep
                      // when the browser is left inactive for a few minutes. This z-index keeps the PDOM from interfering, while still
                      // allowing us to use `fixed`. If the PDOM elements are ever styled with position: 'absolute' (would require
                      // PDOM elements to be positioned relative to focusable ancestors rather than viewport), this could be removed.
                      // See https://github.com/phetsims/joist/blob/master/js/Heartbeat.js as well for the workaround.
                      'z-index: 1;' +

                      // JUST FOR DEBUGGING! So you can see the PDOM on top of other graphical content
                      // 'z-index: 5000;' +

                      // a catch all for things that are not hidden by the styling on descendants of the root (for example
                      // there is no other way to hide or style check boxes with CSS)
                      'opacity: 0.0001;' +
                      '}'
);

SceneryStyle.addRule( `.${LIST_ITEM_CLASS_NAME
                      }{` +

                      // removing list styling prevents a VoiceOver behavior where 'bullet' is read in a confusing way.
                      // Add the LIST_ITEM_CLASS_NAME class with setPDOMClass() to Nodes represented with 'li'
                      // siblings to prevent this. See https://github.com/phetsims/a11y-research/issues/158.
                      'list-style: none;'
);

const PDOMSiblingStyle = {
  SIBLING_CLASS_NAME: SIBLING_CLASS_NAME,
  ROOT_CLASS_NAME: ROOT_CLASS_NAME,
  LIST_ITEM_CLASS_NAME: LIST_ITEM_CLASS_NAME
};

scenery.register( 'PDOMSiblingStyle', PDOMSiblingStyle );
export default PDOMSiblingStyle;