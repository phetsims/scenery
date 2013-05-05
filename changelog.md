
Scenery Changelog
=================

* 2013-5-4  Text nodes can now share Font instances for performance reasons. This means a change to a Font will update
            any Text nodes referring to it. Calling a mutator (like text.fontWeight = 'bold') will internally create a
            copy of the Font instance, and the Text node will not be affected by changes to the previous Font object.
            
* 2013-5-3  Removed Text.textAlign and Text.textBaseline, due to complexity, speed and duplication (with bounds methods)

* 2013-4-29 Removed Text.isHtml flag, added HTMLText instead for HTML-styled text

* 2013-4-27 Added isHtml flag to Text that will treat text as HTML and force the DOM renderer.

* 2013-4-27 Added DOM renderer support for Text (disallows strokes, only allows normal color fill)

* 2013-4-24 Changed input event order for moves: move happens before other associated enter/exit/over/out events,
            and the 'enter' event order is reversed to match DOM events.

* 2013-4-24 Added Text.boundsMethod to switch between text-bounds detection methods

* 2013-4-24 Upgraded documentation and other HTML files to jQuery 2.0.0

* 2013-4-11 Added new scenery.Path( svgPathString )

* 2013-4-5  Changed Circle.circleRadius => Circle.radius

* 2013-4-2  Added Node.center (ES5)

* 2013-3-30 Added toString() serialization to Scene/Node (and subtypes), and Scene.toStringWithChildren()

* 2013-3-30 Added 'matrix' mutator to Node for inline use of Matrix3

* 2013-3-30 Added Node 'opacity' support
