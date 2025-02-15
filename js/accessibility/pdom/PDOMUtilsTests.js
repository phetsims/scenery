// Copyright 2018-2025, University of Colorado Boulder

/**
 * ParallelDOM tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import Node from '../../nodes/Node.js';
import RichText from '../../nodes/RichText.js';
import Text from '../../nodes/Text.js';
import { findStringProperty } from './findStringProperty.js';
import PDOMUtils from './PDOMUtils.js';

QUnit.module( 'AccessibilityUtils' );

// tests
QUnit.test( 'insertElements', assert => {

  const div1 = document.createElement( 'div1' );
  const div2 = document.createElement( 'div2' );
  const div3 = document.createElement( 'div3' );
  const div4 = document.createElement( 'div4' );

  PDOMUtils.insertElements( div1, [ div2, div3, div4 ] );

  assert.ok( div1.childNodes.length === 3, 'inserted number of elements' );
  assert.ok( div1.childNodes[ 0 ] === div2, 'inserted div2 order of elements' );
  assert.ok( div1.childNodes[ 1 ] === div3, 'inserted div3 order of elements' );
  assert.ok( div1.childNodes[ 2 ] === div4, 'inserted div4 order of elements' );


  const div5 = document.createElement( 'div5' );
  const div6 = document.createElement( 'div6' );
  const div7 = document.createElement( 'div7' );

  PDOMUtils.insertElements( div1, [ div5, div6, div7 ], div3 );
  assert.ok( div1.childNodes[ 0 ] === div2, 'inserted div2 order of elements' );
  assert.ok( div1.childNodes[ 1 ] === div5, 'inserted div5 order of elements' );
  assert.ok( div1.childNodes[ 2 ] === div6, 'inserted div6 order of elements' );
  assert.ok( div1.childNodes[ 3 ] === div7, 'inserted div7 order of elements' );
  assert.ok( div1.childNodes[ 4 ] === div3, 'inserted div3 order of elements' );
  assert.ok( div1.childNodes[ 5 ] === div4, 'inserted div4 order of elements' );
} );

QUnit.test( 'getNext/PreviousFocusable', assert => {
  const parent = PDOMUtils.createElement( 'div', false );

  const button = PDOMUtils.createElement( 'button', true ); // focusable
  const div = PDOMUtils.createElement( 'div', true ); // focusable
  const p = PDOMUtils.createElement( 'p', false ); // not focusable

  // elements must be in DOM to be focusable
  document.body.appendChild( parent );
  parent.appendChild( button );
  parent.appendChild( div );
  parent.appendChild( p );

  const firstFocusable = PDOMUtils.getNextFocusable( parent );
  assert.ok( firstFocusable === button, 'first focusable found' );
  firstFocusable.focus();

  const secondFocusable = PDOMUtils.getNextFocusable( parent );
  assert.ok( secondFocusable === div, 'second focusable found' );
  secondFocusable.focus();

  // should still return the div because the p isn't focusable
  const thirdFocusable = PDOMUtils.getNextFocusable( parent );
  assert.ok( thirdFocusable === div, 'no more focusables after div' );

  // remove the DOM nodes so they don't clutter the tests
  document.body.removeChild( parent );
} );

QUnit.test( 'overrideFocusWithTabIndex', assert => {

  // test function directly
  const testButton = document.createElement( 'button' );
  const testListItem = document.createElement( 'li' );
  const testLink = document.createElement( 'a' );
  const testSection = document.createElement( 'section' );

  // defaults, should not an tabindex to any elements
  PDOMUtils.overrideFocusWithTabIndex( testButton, true );
  PDOMUtils.overrideFocusWithTabIndex( testLink, true );
  PDOMUtils.overrideFocusWithTabIndex( testListItem, false );
  PDOMUtils.overrideFocusWithTabIndex( testSection, false );
  assert.ok( testButton.getAttribute( 'tabindex' ) === null, 'testButton focusable by default, shouldn\'t have override' );
  assert.ok( testLink.getAttribute( 'tabindex' ) === null, 'testLink focusable by default, shouldn\'t have override' );
  assert.ok( testListItem.getAttribute( 'tabindex' ) === null, 'testListItem not focusable by default, shouldn\'t have override' );
  assert.ok( testSection.getAttribute( 'tabindex' ) === null, 'testSection not focusable by default, shouldn\'t have override' );

  // override all, should all should have a tabindex
  PDOMUtils.overrideFocusWithTabIndex( testButton, false );
  PDOMUtils.overrideFocusWithTabIndex( testLink, false );
  PDOMUtils.overrideFocusWithTabIndex( testListItem, true );
  PDOMUtils.overrideFocusWithTabIndex( testSection, true );
  assert.ok( testButton.getAttribute( 'tabindex' ) === '-1', 'testButton focusable by default, should have override' );
  assert.ok( testLink.getAttribute( 'tabindex' ) === '-1', 'testLink focusable by default, should have override' );
  assert.ok( testListItem.getAttribute( 'tabindex' ) === '0', 'testListItem not focusable by default, should have override' );
  assert.ok( testSection.getAttribute( 'tabindex' ) === '0', 'testSection not focusable by default, should have override' );

  // test function in usages with createElement
  // tab index should only be set on elements where we are overriding what is being done natively in the browser
  const defaultButton = PDOMUtils.createElement( 'button', true ); // focusable
  const defaultParagraph = PDOMUtils.createElement( 'p', false ); // not focusable
  const defaultDiv = PDOMUtils.createElement( 'div', false ); // not focusable

  // use getAttribute because tabIndex DOM property is always provided by default
  assert.ok( defaultButton.getAttribute( 'tabindex' ) === null, 'default button has no tab index' );
  assert.ok( defaultParagraph.getAttribute( 'tabindex' ) === null, 'default paragraph has no tab index' );
  assert.ok( defaultDiv.getAttribute( 'tabindex' ) === null, 'default div has no tab index' );

  // custom focusability should all have tab indices, even those that are being removed from the document
  const customButton = PDOMUtils.createElement( 'button', false ); // not focusable
  const customParagraph = PDOMUtils.createElement( 'p', true ); // focusable
  const customDiv = PDOMUtils.createElement( 'div', true ); // focusable

  assert.ok( customButton.getAttribute( 'tabindex' ) === '-1', 'custom button removed from focus' );
  assert.ok( customParagraph.getAttribute( 'tabindex' ) === '0', 'custom paragraph added to focus' );
  assert.ok( customDiv.getAttribute( 'tabindex' ) === '0', 'custom button removed from focus' );
} );

QUnit.test( 'setTextContent', assert => {
  const toyElement = PDOMUtils.createElement( 'div' );

  // basic string
  const stringContent = 'I am feeling pretty flat today.';

  // formatted content
  const htmlContent = 'I am <i>feeling</i> rather <strong>BOLD</strong> today';

  // malformed formatting tags
  const malformedHTMLContent = 'I am feeling a <b>bit off> today.';

  // tags not allowed as innerHTML
  const invalidHTMLContent = 'I am feeling a bit <a href="daring">devious</a> today.';

  PDOMUtils.setTextContent( toyElement, stringContent );
  assert.ok( toyElement.textContent === stringContent, 'textContent set for basic string' );
  assert.ok( toyElement.firstElementChild === null, 'no innerHTML for basic string' );

  PDOMUtils.setTextContent( toyElement, htmlContent );
  assert.ok( toyElement.innerHTML === htmlContent, 'innerHTML set for formated content' );
  assert.ok( toyElement.firstElementChild !== null, 'element child exists for innerHTML' );

  PDOMUtils.setTextContent( toyElement, malformedHTMLContent );
  assert.ok( toyElement.textContent === malformedHTMLContent, 'malformed HTML set as content' );
  assert.ok( toyElement.firstElementChild === null, 'fallback to textContent for malformed tags' );

  PDOMUtils.setTextContent( toyElement, invalidHTMLContent );
  assert.ok( toyElement.textContent === invalidHTMLContent, 'invalid HTML set as content' );
  assert.ok( toyElement.firstElementChild === null, 'fallback to textContent for disallowed tags' );
} );

QUnit.test( 'findStringProperty', assert => {
  const testString = 'testing';

  const rootNode = new Node();
  const a = new Node();
  const b = new Node();
  const testText = new Text( testString );
  const testRichText = new RichText( testString );

  rootNode.addChild( a );
  a.addChild( b );
  b.addChild( testText );

  // basic find test
  let foundStringProperty = findStringProperty( rootNode );
  assert.ok( foundStringProperty && foundStringProperty.value === testString, 'found the string content' );

  // test with no string to find
  b.removeChild( testText );
  foundStringProperty = findStringProperty( rootNode );
  assert.ok( foundStringProperty === null, 'no string content found' );

  // test with RichText
  b.addChild( testRichText );
  foundStringProperty = findStringProperty( rootNode );
  assert.ok( foundStringProperty && foundStringProperty.value === testString, 'found the RichText content' );

  // test with an empty Node
  foundStringProperty = findStringProperty( new Node() );
  assert.ok( foundStringProperty === null, 'no content found in empty Node' );

  // test with Text and RichText in the subtree
  b.addChild( testText );
  foundStringProperty = findStringProperty( rootNode );
  assert.ok( foundStringProperty && foundStringProperty.value === testString, 'found the string content in subtree' );
} );