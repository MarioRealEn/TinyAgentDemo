tell application "Notes"
    set demoAccount to default account
    tell demoAccount
        if not (exists folder "TinyAgent Demo") then
            make new folder with properties {name:"TinyAgent Demo"}
        end if
        tell folder "TinyAgent Demo"
            if exists note "Demo brief" then
                set brief to first note whose name is "Demo brief"
            else
                set brief to make new note with properties {body:"<h1>Demo brief</h1><p>We are presenting our local AI assistant tomorrow. Before presenting, we need to prepare the slide deck, charge the laptop, test the projector adapter, print five one-page handouts, and rehearse the three-minute introduction.</p>"}
            end if
            show brief
            return id of brief
        end tell
    end tell
end tell
